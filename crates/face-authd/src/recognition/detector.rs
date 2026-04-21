use std::ffi::CString;

use anyhow::{bail, Context, Result};

use crate::camera::GrayscaleFrame;
use super::ffi;
use super::models::{ensure_model, SP5_MODEL, REC_MODEL};

/// A detected face: bounding box and 5 landmarks (dlib 5-point predictor).
#[derive(Debug, Clone)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    #[allow(dead_code)]
    pub landmarks: [[f32; 2]; 5],
    pub score: f32,
}

pub struct FaceDetector {
    ctx: *mut ffi::DlibCtx,
}

unsafe impl Send for FaceDetector {}

impl FaceDetector {
    /// Load just the HOG detector + shape predictor (no rec model needed).
    pub fn load() -> Result<Self> {
        let sp_path = ensure_model(&SP5_MODEL)?;
        let sp_cstr = CString::new(sp_path.to_str().context("sp5 path not UTF-8")?)
            .context("sp5 path contains nul")?;

        let ctx = unsafe { ffi::dlib_create(sp_cstr.as_ptr(), std::ptr::null()) };
        if ctx.is_null() {
            bail!("dlib_create failed: {}", last_error());
        }
        Ok(Self { ctx })
    }

    /// Load HOG detector + shape predictor + rec model (shared with FaceRecognizer).
    pub(super) fn load_with_rec(sp_path: &std::path::Path, rec_path: &std::path::Path) -> Result<Self> {
        let sp_cstr = CString::new(sp_path.to_str().context("sp5 path not UTF-8")?)
            .context("sp5 path contains nul")?;
        let rec_cstr = CString::new(rec_path.to_str().context("rec path not UTF-8")?)
            .context("rec path contains nul")?;

        let ctx = unsafe { ffi::dlib_create(sp_cstr.as_ptr(), rec_cstr.as_ptr()) };
        if ctx.is_null() {
            bail!("dlib_create failed: {}", last_error());
        }
        Ok(Self { ctx })
    }

    pub(super) fn raw_ctx(&mut self) -> *mut ffi::DlibCtx {
        self.ctx
    }

    /// Detect faces. Returns detections sorted by score descending.
    pub fn detect(&mut self, frame: &GrayscaleFrame) -> Result<Vec<Detection>> {
        detect_on_ctx(self.ctx, frame)
    }
}

impl Drop for FaceDetector {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ffi::dlib_destroy(self.ctx) };
        }
    }
}

/// Run detection on an arbitrary ctx (used by FaceRecognizer too).
pub(super) fn detect_on_ctx(ctx: *mut ffi::DlibCtx, frame: &GrayscaleFrame) -> Result<Vec<Detection>> {
    const MAX_FACES: usize = 16;
    let mut rects  = [0f32; MAX_FACES * 4];
    let mut scores = [0f32; MAX_FACES];
    let mut lms    = [0f32; MAX_FACES * 10];

    let n = unsafe {
        ffi::dlib_detect(
            ctx,
            frame.pixels.as_ptr(),
            frame.width as i32,
            frame.height as i32,
            rects.as_mut_ptr(),
            scores.as_mut_ptr(),
            lms.as_mut_ptr(),
            MAX_FACES as i32,
        )
    };

    if n < 0 {
        bail!("dlib_detect failed: {}", last_error());
    }

    let mut detections = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        let b = i * 10;
        detections.push(Detection {
            x: rects[i*4],
            y: rects[i*4+1],
            w: rects[i*4+2],
            h: rects[i*4+3],
            score: scores[i],
            landmarks: [
                [lms[b],   lms[b+1]],
                [lms[b+2], lms[b+3]],
                [lms[b+4], lms[b+5]],
                [lms[b+6], lms[b+7]],
                [lms[b+8], lms[b+9]],
            ],
        });
    }

    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    Ok(detections)
}

pub(super) fn last_error() -> String {
    unsafe {
        std::ffi::CStr::from_ptr(ffi::dlib_last_error())
            .to_string_lossy()
            .into_owned()
    }
}

pub(super) fn load_models() -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let sp = ensure_model(&SP5_MODEL)?;
    let rec = ensure_model(&REC_MODEL)?;
    Ok((sp, rec))
}
