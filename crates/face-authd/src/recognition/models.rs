use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

pub struct ModelSpec {
    /// Final (decompressed) filename stored on disk.
    pub filename: &'static str,
    /// Download URL; if it ends with `.bz2` the file is decompressed after download.
    pub url: &'static str,
    /// Hex-encoded SHA-256 of the *decompressed* file; empty = skip verification.
    pub sha256: &'static str,
}

/// dlib 5-point face landmark predictor.
pub const SP5_MODEL: ModelSpec = ModelSpec {
    filename: "shape_predictor_5_face_landmarks.dat",
    url: "https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2",
    sha256: "",
};

/// dlib ResNet face recognition model (128-dim descriptor).
pub const REC_MODEL: ModelSpec = ModelSpec {
    filename: "dlib_face_recognition_resnet_model_v1.dat",
    url: "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    sha256: "",
};

pub fn model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("FACE_AUTHD_MODEL_DIR") {
        return PathBuf::from(dir);
    }

    PathBuf::from("/var/lib/face-authd/models")
}

/// Return the path to a model, downloading and decompressing it if necessary.
pub fn ensure_model(spec: &ModelSpec) -> Result<PathBuf> {
    let dir = model_dir();
    let path = dir.join(spec.filename);

    if path.exists() {
        if !spec.sha256.is_empty() {
            verify_sha256(&path, spec.sha256).with_context(|| {
                format!(
                    "model {} failed checksum; delete it to re-download",
                    path.display()
                )
            })?;
        }
        return Ok(path);
    }

    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create model directory {}", dir.display()))?;

    eprintln!("Downloading {} → {}", spec.filename, path.display());
    let compressed = download_to_memory(spec.url)
        .with_context(|| format!("failed to download {}", spec.url))?;

    let data = if spec.url.ends_with(".bz2") {
        decompress_bz2(&compressed)
            .with_context(|| format!("failed to decompress {}", spec.url))?
    } else {
        compressed
    };

    let tmp = path.with_extension("tmp");
    fs::write(&tmp, &data)
        .with_context(|| format!("failed to write {}", tmp.display()))?;

    if !spec.sha256.is_empty() {
        verify_sha256_bytes(&data, spec.sha256)
            .map_err(|err| {
                let _ = fs::remove_file(&tmp);
                err
            })
            .with_context(|| format!("downloaded {} failed checksum; removed", spec.filename))?;
    }

    fs::rename(&tmp, &path)
        .with_context(|| format!("failed to rename {} → {}", tmp.display(), path.display()))?;

    eprintln!("Saved {} ({:.1} MB)", spec.filename, data.len() as f64 / 1_048_576.0);
    Ok(path)
}

fn download_to_memory(url: &str) -> Result<Vec<u8>> {
    let mut response = reqwest::blocking::get(url)
        .with_context(|| format!("HTTP GET failed for {url}"))?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP {} for {url}", response.status());
    }

    let total = response.content_length();
    let mut data = Vec::with_capacity(total.unwrap_or(0) as usize);
    let mut buf = vec![0u8; 65536];

    loop {
        let n = std::io::Read::read(&mut response, &mut buf)
            .context("read error during download")?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
        match total {
            Some(t) => eprint!("\r  {:.1} / {:.1} MB", mb(data.len() as u64), mb(t)),
            None    => eprint!("\r  {:.1} MB", mb(data.len() as u64)),
        }
    }
    let _ = std::io::stderr().flush();
    eprintln!();
    Ok(data)
}

fn decompress_bz2(data: &[u8]) -> Result<Vec<u8>> {
    use bzip2::read::BzDecoder;
    use std::io::Read;

    let mut decoder = BzDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).context("bz2 decompression failed")?;
    Ok(out)
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let data = fs::read(path)
        .with_context(|| format!("cannot read {} for checksum", path.display()))?;
    verify_sha256_bytes(&data, expected)
}

fn verify_sha256_bytes(data: &[u8], expected: &str) -> Result<()> {
    let actual = format!("{:x}", Sha256::digest(data));
    if actual != expected {
        anyhow::bail!("checksum mismatch: expected {} got {}", expected, actual);
    }
    Ok(())
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1_048_576.0
}
