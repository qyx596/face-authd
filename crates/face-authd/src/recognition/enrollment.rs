use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use super::embedder::{valid_embedding, Embedding};
use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result};
use base64::Engine;
use common::protocol::valid_username;
use keyutils::{keytypes, Keyring, SpecialKeyring};
use serde::{Deserialize, Serialize};

pub const MODEL_ID: &str = "yunet2023-sface2021-unit128-bgr-v1";
const LEGACY_MODEL_ID: &str = "dlib-resnet-v1-raw128-gray";
fn default_model_id() -> String {
    LEGACY_MODEL_ID.into()
}
#[derive(Debug, Serialize, Deserialize)]
pub struct UserEnrollment {
    pub username: String,
    #[serde(default = "default_model_id")]
    pub model_id: String,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EncryptedEnrollment {
    format: String,
    nonce_b64: String,
    ciphertext_b64: String,
}
#[cfg(test)]
const FORMAT_V1: &str = "face-authd-enrollment-v1";
const FORMAT_V3: &str = "face-authd-enrollment-v3-sface";
const KEY_FILE: &str = "enrollment.key";

impl UserEnrollment {
    pub fn embeddings_as_arrays(&self) -> Vec<Embedding> {
        self.embeddings
            .iter()
            .filter_map(|e| {
                let a: Embedding = e.as_slice().try_into().ok()?;
                valid_embedding(&a).then_some(a)
            })
            .collect()
    }
    fn validate(&self, username: &str) -> Result<()> {
        anyhow::ensure!(self.username == username, "template username mismatch");
        anyhow::ensure!(
            self.model_id == MODEL_ID,
            "incompatible model; enroll again"
        );
        anyhow::ensure!(
            !self.embeddings.is_empty()
                && self.embeddings.len() <= 100
                && self.embeddings_as_arrays().len() == self.embeddings.len(),
            "invalid enrollment descriptors"
        );
        Ok(())
    }
}

pub fn enrollment_dir() -> PathBuf {
    std::env::var_os("FACE_AUTHD_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/lib/face-authd"))
}
pub fn enrollment_path(username: &str) -> PathBuf {
    enrollment_dir().join(username).join("enrollment.json")
}

fn check_username(username: &str) -> Result<()> {
    anyhow::ensure!(valid_username(username), "invalid username");
    Ok(())
}
fn secure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        use std::os::unix::fs::DirBuilderExt;
        fs::DirBuilder::new()
            .mode(0o700)
            .recursive(true)
            .create(path)?;
    }
    let m = fs::symlink_metadata(path)?;
    anyhow::ensure!(
        m.is_dir() && !m.file_type().is_symlink() && m.uid() == unsafe { libc::geteuid() },
        "untrusted enrollment directory {}",
        path.display()
    );
    anyhow::ensure!(
        m.mode() & 0o022 == 0,
        "enrollment directory is writable by other users"
    );
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}
pub fn check_enrollment_dir(path: &Path) -> Result<()> {
    secure_dir(path)
}
fn open_private(path: &Path) -> Result<File> {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    let m = f.metadata()?;
    anyhow::ensure!(
        m.is_file() && m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o077 == 0,
        "unsafe owner or permissions on {} (expected owner-only access)",
        path.display()
    );
    Ok(f)
}
fn lock_store(dir: &Path) -> Result<File> {
    secure_dir(dir)?;
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(dir.join(".enrollment.lock"))?;
    anyhow::ensure!(
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0,
        "failed to lock enrollment store"
    );
    Ok(file)
}
fn atomic_private_write(path: &Path, data: &[u8]) -> Result<()> {
    let parent = path.parent().context("file has no parent")?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)?; // Created with 0600, unique name.
    temp.write_all(data)?;
    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|e| e.error)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}
fn read_key(dir: &Path, create: bool) -> Result<[u8; 32]> {
    let path = dir.join(KEY_FILE);
    if path.try_exists()? {
        let mut key = Vec::new();
        open_private(&path)?.take(33).read_to_end(&mut key)?;
        return key
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid enrollment key size"));
    }
    // Migrate the original keyring key, if present, without changing its value.
    if let Ok(mut session) = Keyring::attach_or_create(SpecialKeyring::Session) {
        if let Ok(keyring) = session.attach_persistent() {
            if let Ok(key) =
                keyring.search_for_key::<keytypes::User, _, _>("face-authd:enrollment-key:v1", None)
            {
                let key: [u8; 32] = key
                    .read()?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid legacy key size"))?;
                atomic_private_write(&path, &key)?;
                return Ok(key);
            }
        }
    }
    anyhow::ensure!(
        create,
        "enrollment key missing; restore the original key or explicitly re-enroll"
    );
    let mut key = [0; 32];
    File::open("/dev/urandom")?.read_exact(&mut key)?;
    atomic_private_write(&path, &key)?;
    Ok(key)
}
fn aad(username: &str) -> Vec<u8> {
    format!("{FORMAT_V3}:{username}:{MODEL_ID}").into_bytes()
}
fn encrypt_blob(key: &[u8; 32], username: &str, plaintext: &[u8]) -> Result<EncryptedEnrollment> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| anyhow::anyhow!("invalid key"))?;
    let mut nonce = [0u8; 12];
    File::open("/dev/urandom")?.read_exact(&mut nonce)?;
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext,
                aad: &aad(username),
            },
        )
        .map_err(|_| anyhow::anyhow!("encryption failed"))?;
    Ok(EncryptedEnrollment {
        format: FORMAT_V3.into(),
        nonce_b64: base64::engine::general_purpose::STANDARD.encode(nonce),
        ciphertext_b64: base64::engine::general_purpose::STANDARD.encode(ciphertext),
    })
}
fn decrypt_blob(key: &[u8; 32], username: &str, sealed: &EncryptedEnrollment) -> Result<Vec<u8>> {
    anyhow::ensure!(
        sealed.format == FORMAT_V3,
        "legacy or unsupported enrollment; YuNet/SFace requires re-enrollment with --replace"
    );
    let nonce = base64::engine::general_purpose::STANDARD.decode(&sealed.nonce_b64)?;
    anyhow::ensure!(nonce.len() == 12, "invalid nonce size");
    let ciphertext = base64::engine::general_purpose::STANDARD.decode(&sealed.ciphertext_b64)?;
    let associated = if sealed.format == FORMAT_V3 {
        aad(username)
    } else {
        Vec::new()
    };
    Aes256Gcm::new_from_slice(key)
        .map_err(|_| anyhow::anyhow!("invalid key"))?
        .decrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &ciphertext,
                aad: &associated,
            },
        )
        .map_err(|_| anyhow::anyhow!("template authentication failed"))
}
fn load_at(dir: &Path, username: &str) -> Result<Option<UserEnrollment>> {
    check_username(username)?;
    if !dir.try_exists()? {
        return Ok(None);
    }
    let _lock = lock_store(dir)?;
    let parent = dir.join(username);
    if !parent.try_exists()? {
        return Ok(None);
    }
    secure_dir(&parent)?;
    let path = parent.join("enrollment.json");
    if !path.try_exists()? {
        return Ok(None);
    }
    let mut raw = Vec::new();
    open_private(&path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    anyhow::ensure!(raw.len() <= 1024 * 1024, "enrollment too large");
    let sealed: EncryptedEnrollment = serde_json::from_slice(&raw)
        .context("unencrypted or invalid template; explicitly re-enroll with --replace")?;
    anyhow::ensure!(
        sealed.format == FORMAT_V3,
        "legacy enrollment; YuNet/SFace requires re-enrollment with --replace"
    );
    let key = read_key(dir, false)?;
    let enrollment: UserEnrollment =
        serde_json::from_slice(&decrypt_blob(&key, username, &sealed)?)?;
    enrollment.validate(username)?;
    Ok(Some(enrollment))
}
fn save_at(dir: &Path, enrollment: &UserEnrollment) -> Result<()> {
    check_username(&enrollment.username)?;
    enrollment.validate(&enrollment.username)?;
    let _lock = lock_store(dir)?;
    let parent = dir.join(&enrollment.username);
    secure_dir(&parent)?;
    let key = read_key(dir, true)?;
    let sealed = encrypt_blob(&key, &enrollment.username, &serde_json::to_vec(enrollment)?)?;
    atomic_private_write(
        &parent.join("enrollment.json"),
        &serde_json::to_vec_pretty(&sealed)?,
    )
}
pub fn load_enrollment(username: &str) -> Result<Option<UserEnrollment>> {
    load_at(&enrollment_dir(), username)
}
pub fn save_enrollment(enrollment: &UserEnrollment) -> Result<()> {
    save_at(&enrollment_dir(), enrollment)
}
pub fn delete_enrollment(username: &str) -> Result<bool> {
    check_username(username)?;
    let dir = enrollment_dir();
    if !dir.try_exists()? {
        return Ok(false);
    }
    let _lock = lock_store(&dir)?;
    let parent = dir.join(username);
    if !parent.try_exists()? {
        return Ok(false);
    }
    secure_dir(&parent)?;
    match fs::remove_file(parent.join("enrollment.json")) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e.into()),
    }
}
pub fn list_enrolled_users() -> Result<Vec<String>> {
    let dir = enrollment_dir();
    if !dir.try_exists()? {
        return Ok(Vec::new());
    }
    let _lock = lock_store(&dir)?;
    let mut users = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if valid_username(&name)
            && entry.file_type()?.is_dir()
            && entry.path().join("enrollment.json").exists()
        {
            users.push(name);
        }
    }
    users.sort();
    Ok(users)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn enrollment(name: &str) -> UserEnrollment {
        UserEnrollment {
            username: name.into(),
            model_id: MODEL_ID.into(),
            embeddings: vec![vec![1.0_f32 / 128.0_f32.sqrt(); 128]],
        }
    }
    #[test]
    fn encrypted_roundtrip_and_private_permissions() {
        let dir = tempfile::tempdir().unwrap();
        save_at(dir.path(), &enrollment("alice")).unwrap();
        let path = dir.path().join("alice/enrollment.json");
        assert_eq!(fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert_eq!(
            fs::metadata(dir.path().join(KEY_FILE)).unwrap().mode() & 0o777,
            0o600
        );
        assert_eq!(
            load_at(dir.path(), "alice").unwrap().unwrap().username,
            "alice"
        );
        let sealed = fs::read(&path).unwrap();
        secure_dir(&dir.path().join("bob")).unwrap();
        fs::write(dir.path().join("bob/enrollment.json"), &sealed).unwrap();
        fs::set_permissions(
            dir.path().join("bob/enrollment.json"),
            fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        assert!(load_at(dir.path(), "bob").is_err());
    }
    #[test]
    fn tampering_and_wrong_key_are_rejected() {
        let mut sealed = encrypt_blob(&[1; 32], "alice", b"private").unwrap();
        assert!(decrypt_blob(&[2; 32], "alice", &sealed).is_err());
        assert!(decrypt_blob(&[1; 32], "bob", &sealed).is_err());
        sealed.nonce_b64 = "AA==".into();
        assert!(decrypt_blob(&[1; 32], "alice", &sealed).is_err());
    }
    #[test]
    fn legacy_encrypted_templates_require_reenrollment() {
        let key = [1; 32];
        let nonce = [2; 12];
        let cipher = Aes256Gcm::new_from_slice(&key).unwrap();
        let data = serde_json::to_vec(&enrollment("alice")).unwrap();
        let sealed = EncryptedEnrollment {
            format: FORMAT_V1.into(),
            nonce_b64: base64::engine::general_purpose::STANDARD.encode(nonce),
            ciphertext_b64: base64::engine::general_purpose::STANDARD.encode(
                cipher
                    .encrypt(Nonce::from_slice(&nonce), data.as_slice())
                    .unwrap(),
            ),
        };
        assert!(decrypt_blob(&key, "alice", &sealed).is_err());
        let legacy = r#"{"username":"alice","embeddings":[[1.0]]}"#;
        let e: UserEnrollment = serde_json::from_str(legacy).unwrap();
        assert_eq!(e.model_id, LEGACY_MODEL_ID);
        assert!(e.validate("alice").is_err());
    }
    #[test]
    fn symlink_and_missing_key_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        save_at(dir.path(), &enrollment("alice")).unwrap();
        fs::remove_file(dir.path().join(KEY_FILE)).unwrap();
        assert!(load_at(dir.path(), "alice").is_err());
        assert!(!dir.path().join(KEY_FILE).exists());
        std::os::unix::fs::symlink("alice", dir.path().join("bob")).unwrap();
        assert!(save_at(dir.path(), &enrollment("bob")).is_err());
        assert!(save_at(dir.path(), &enrollment("../escape")).is_err());
    }
    #[test]
    fn simultaneous_enrollments_share_one_key() {
        let dir = tempfile::tempdir().unwrap();
        std::thread::scope(|s| {
            let a = s.spawn(|| save_at(dir.path(), &enrollment("alice")));
            let b = s.spawn(|| save_at(dir.path(), &enrollment("bob")));
            a.join().unwrap().unwrap();
            b.join().unwrap().unwrap();
        });
        assert!(load_at(dir.path(), "alice").unwrap().is_some());
        assert!(load_at(dir.path(), "bob").unwrap().is_some());
    }
}
