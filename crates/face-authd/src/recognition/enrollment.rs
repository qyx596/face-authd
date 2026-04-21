use std::fs;
use std::path::{Path, PathBuf};
use std::os::unix::fs::PermissionsExt;

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result};
use base64::Engine;
use keyutils::{keytypes, Keyring, SpecialKeyring};
use serde::{Deserialize, Serialize};

use super::embedder::Embedding;

#[derive(Debug, Serialize, Deserialize)]
pub struct UserEnrollment {
    pub username: String,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EncryptedEnrollment {
    format: String,
    nonce_b64: String,
    ciphertext_b64: String,
}

const ENCRYPTED_FORMAT_V1: &str = "face-authd-enrollment-v1";
const KEYRING_KEY_DESC: &str = "face-authd:enrollment-key:v1";
const NONCE_SIZE: usize = 12;

impl UserEnrollment {
    pub fn embeddings_as_arrays(&self) -> Vec<Embedding> {
        self.embeddings
            .iter()
            .filter_map(|e| {
                if e.len() == 128 {
                    let mut arr = [0f32; 128];
                    arr.copy_from_slice(e);
                    Some(arr)
                } else {
                    None
                }
            })
            .collect()
    }
}

pub fn enrollment_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FACE_AUTHD_DATA_DIR") {
        return PathBuf::from(dir);
    }
    // Default to system-wide storage so daemon and CLI share one enrollment set.
    PathBuf::from("/var/lib/face-authd")
}

pub fn enrollment_path(username: &str) -> PathBuf {
    enrollment_dir()
        .join(sanitize_username(username))
        .join("enrollment.json")
}

pub fn load_enrollment(username: &str) -> Result<Option<UserEnrollment>> {
    let path = enrollment_path(username);
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read(&path)
        .with_context(|| format!("failed to read enrollment from {}", path.display()))?;

    // Backward-compatible: support legacy plaintext enrollment JSON.
    if let Ok(enrollment) = serde_json::from_slice::<UserEnrollment>(&raw) {
        return Ok(Some(enrollment));
    }

    let encrypted: EncryptedEnrollment = serde_json::from_slice(&raw)
        .with_context(|| format!("failed to parse enrollment envelope at {}", path.display()))?;
    if encrypted.format != ENCRYPTED_FORMAT_V1 {
        anyhow::bail!(
            "unsupported enrollment format '{}' at {}",
            encrypted.format,
            path.display()
        );
    }
    let key = read_or_create_key()?;
    let plaintext = decrypt_blob(&key, &encrypted)
        .with_context(|| format!("failed to decrypt enrollment at {}", path.display()))?;
    let enrollment: UserEnrollment = serde_json::from_slice(&plaintext)
        .with_context(|| format!("failed to parse decrypted enrollment at {}", path.display()))?;
    Ok(Some(enrollment))
}

pub fn save_enrollment(enrollment: &UserEnrollment) -> Result<()> {
    let path = enrollment_path(&enrollment.username);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    let key = read_or_create_key()?;
    let plain = serde_json::to_vec(enrollment).context("failed to serialize enrollment")?;
    let sealed = encrypt_blob(&key, &plain).context("failed to encrypt enrollment")?;
    let data = serde_json::to_vec_pretty(&sealed).context("failed to serialize encrypted enrollment")?;
    // Write atomically via temp file
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, data)
        .with_context(|| format!("failed to write enrollment to {}", tmp.display()))?;
    fs::rename(&tmp, &path)
        .with_context(|| format!("failed to rename enrollment to {}", path.display()))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("failed to set permissions on {}", path.display()))?;
    Ok(())
}

pub fn delete_enrollment(username: &str) -> Result<bool> {
    let path = enrollment_path(username);
    if !path.exists() {
        return Ok(false);
    }
    fs::remove_file(&path)
        .with_context(|| format!("failed to delete enrollment at {}", path.display()))?;
    Ok(true)
}

pub fn list_enrolled_users() -> Result<Vec<String>> {
    let dir = enrollment_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut users = Vec::new();
    for entry in fs::read_dir(&dir)
        .with_context(|| format!("failed to read enrollment directory {}", dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let json_path = entry.path().join("enrollment.json");
        if json_path.exists() {
            users.push(name);
        }
    }
    users.sort();
    Ok(users)
}

fn sanitize_username(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect()
}

/// Check the enrollment dir exists and is writable.
pub fn check_enrollment_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("failed to create enrollment directory {}", path.display()))?;
    let test = path.join(".write_test");
    fs::write(&test, b"").with_context(|| format!("enrollment directory {} is not writable", path.display()))?;
    fs::remove_file(&test).ok();
    Ok(())
}

fn read_or_create_key() -> Result<[u8; 32]> {
    let mut session = Keyring::attach_or_create(SpecialKeyring::Session)
        .context("failed to attach session keyring for enrollment key")?;
    let mut keyring = session
        .attach_persistent()
        .context("failed to attach persistent keyring for enrollment key")?;

    if let Ok(key) = keyring.search_for_key::<keytypes::User, _, _>(KEYRING_KEY_DESC, None) {
        let payload = key
            .read()
            .context("failed to read enrollment key from keyring")?;
        return as_fixed_key(&payload);
    }

    let mut key = [0u8; 32];
    let mut urandom = fs::File::open("/dev/urandom")
        .context("failed to open /dev/urandom for key generation")?;
    use std::io::Read;
    urandom
        .read_exact(&mut key)
        .context("failed to read random bytes for enrollment key")?;

    keyring
        .add_key::<keytypes::User, _, _>(KEYRING_KEY_DESC, key.as_slice())
        .context("failed to store enrollment key in keyring")?;
    Ok(key)
}

fn as_fixed_key(decoded: &[u8]) -> Result<[u8; 32]> {
    if decoded.len() != 32 {
        anyhow::bail!("invalid enrollment key size in keyring: expected 32 bytes, got {}", decoded.len());
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&decoded);
    Ok(out)
}

fn encrypt_blob(key: &[u8; 32], plaintext: &[u8]) -> Result<EncryptedEnrollment> {
    let cipher = Aes256Gcm::new_from_slice(key).context("invalid encryption key")?;

    let mut nonce = [0u8; NONCE_SIZE];
    let mut urandom = fs::File::open("/dev/urandom").context("failed to open /dev/urandom")?;
    use std::io::Read;
    urandom.read_exact(&mut nonce).context("failed to generate nonce")?;

    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce), plaintext)
        .map_err(|_| anyhow::anyhow!("encryption failed"))?;

    Ok(EncryptedEnrollment {
        format: ENCRYPTED_FORMAT_V1.to_string(),
        nonce_b64: base64::engine::general_purpose::STANDARD.encode(nonce),
        ciphertext_b64: base64::engine::general_purpose::STANDARD.encode(ciphertext),
    })
}

fn decrypt_blob(key: &[u8; 32], sealed: &EncryptedEnrollment) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key).context("invalid encryption key")?;
    let nonce = base64::engine::general_purpose::STANDARD
        .decode(&sealed.nonce_b64)
        .context("invalid nonce encoding")?;
    if nonce.len() != NONCE_SIZE {
        anyhow::bail!("invalid nonce size: expected {NONCE_SIZE}, got {}", nonce.len());
    }
    let ciphertext = base64::engine::general_purpose::STANDARD
        .decode(&sealed.ciphertext_b64)
        .context("invalid ciphertext encoding")?;
    let plain = cipher
        .decrypt(Nonce::from_slice(&nonce), ciphertext.as_ref())
        .map_err(|_| anyhow::anyhow!("decryption failed"))?;
    Ok(plain)
}
