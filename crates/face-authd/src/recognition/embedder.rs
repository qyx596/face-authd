use anyhow::{bail, Result};

use crate::camera::GrayscaleFrame;
use super::detector::{Detection, last_error};
use super::ffi;

/// 128-dimensional L2-normalised face descriptor (dlib face recognition model).
pub type Embedding = [f32; 128];

pub(super) fn embed_on_ctx(
    ctx: *mut ffi::DlibCtx,
    frame: &GrayscaleFrame,
    det: &Detection,
) -> Result<Embedding> {
    let mut emb = [0f32; 128];

    let ret = unsafe {
        ffi::dlib_embed(
            ctx,
            frame.pixels.as_ptr(),
            frame.width as i32,
            frame.height as i32,
            det.x, det.y, det.w, det.h,
            emb.as_mut_ptr(),
        )
    };

    if ret != 0 {
        bail!("dlib_embed failed: {}", last_error());
    }

    Ok(emb)
}

/// Cosine similarity between two L2-normalised embeddings (result in [-1, 1]).
pub fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
