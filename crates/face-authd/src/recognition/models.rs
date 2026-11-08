use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub struct ModelSpec {
    pub filename: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
}
// OpenCV Zoo revision pinned; hashes equal upstream Git LFS object IDs.
pub const DET_MODEL: ModelSpec = ModelSpec {
    filename: "face_detection_yunet_2023mar.onnx",
    url: "https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
};
pub const REC_MODEL: ModelSpec = ModelSpec {
    filename: "face_recognition_sface_2021dec.onnx",
    url: "https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    sha256: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
};
const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024;
pub fn model_dir() -> PathBuf {
    std::env::var_os("FACE_AUTHD_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/var/lib/face-authd/models"))
}
fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    anyhow::ensure!(expected.len() == 64, "model checksum must be pinned");
    let mut file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let m = file.metadata()?;
    anyhow::ensure!(
        m.is_file()
            && m.mode() & 0o022 == 0
            && (m.uid() == 0 || m.uid() == unsafe { libc::geteuid() }),
        "untrusted model file {}",
        path.display()
    );
    anyhow::ensure!(m.len() <= MAX_MODEL_BYTES, "model exceeds size limit");
    let mut sha = Sha256::new();
    std::io::copy(&mut file, &mut sha)?;
    anyhow::ensure!(
        format!("{:x}", sha.finalize()) == expected,
        "model checksum mismatch: {}",
        path.display()
    );
    Ok(())
}
/// Authentication never downloads models or accepts unverified cached weights.
pub fn ensure_model(spec: &ModelSpec) -> Result<PathBuf> {
    let path = model_dir().join(spec.filename);
    verify_sha256(&path, spec.sha256).with_context(|| {
        format!(
            "model {} is missing or invalid; run `face-authd fetch-models` explicitly",
            path.display()
        )
    })?;
    Ok(path)
}
pub fn fetch_models() -> Result<()> {
    let dir = model_dir();
    fs::create_dir_all(&dir)?;
    let meta = fs::symlink_metadata(&dir)?;
    anyhow::ensure!(
        meta.is_dir()
            && !meta.file_type().is_symlink()
            && meta.mode() & 0o022 == 0
            && meta.uid() == unsafe { libc::geteuid() },
        "untrusted model directory"
    );
    let client = reqwest::blocking::Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(90))
        .build()?;
    for spec in [&DET_MODEL, &REC_MODEL] {
        let path = dir.join(spec.filename);
        if verify_sha256(&path, spec.sha256).is_ok() {
            continue;
        }
        let response = client.get(spec.url).send()?.error_for_status()?;
        let mut temp = tempfile::NamedTempFile::new_in(&dir)?;
        let bytes = std::io::copy(&mut response.take(MAX_MODEL_BYTES + 1), &mut temp)?;
        anyhow::ensure!(bytes <= MAX_MODEL_BYTES, "model download exceeds limit");
        temp.flush()?;
        temp.as_file().sync_all()?;
        verify_sha256(temp.path(), spec.sha256)?;
        temp.persist(&path).map_err(|e| e.error)?;
        File::open(&dir)?.sync_all()?;
        eprintln!("Verified {}", spec.filename);
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packaging_uses_the_same_pinned_models() {
        let script = include_str!("../../../../scripts/download-models.sh");
        for model in [&DET_MODEL, &REC_MODEL] {
            assert!(script.contains(model.filename));
            assert!(script.contains(model.sha256));
            let revision = model.url.split('/').nth(6).unwrap();
            assert!(script.contains(revision));
        }
    }
    #[test]
    fn reject_corrupt_and_symlinked_models() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model");
        fs::write(&file, b"weights").unwrap();
        let hash = format!("{:x}", Sha256::digest(b"weights"));
        assert!(verify_sha256(&file, &hash).is_ok());
        assert!(verify_sha256(&file, &"0".repeat(64)).is_err());
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        assert!(verify_sha256(&link, &hash).is_err());
    }
}
