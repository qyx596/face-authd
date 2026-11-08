use super::ffi;
use super::models::{ensure_model, DET_MODEL, REC_MODEL};
use crate::camera::{BgrFrame, GrayscaleFrame};
use anyhow::{bail, Context, Result};
use std::ffi::CString;

/// YuNet's box, five alignment landmarks, and confidence.
#[derive(Debug, Clone)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub landmarks: [[f32; 2]; 5],
    pub score: f32,
}
impl Detection {
    pub(super) fn row(&self) -> [f32; 15] {
        let mut row = [0.0; 15];
        row[..4].copy_from_slice(&[self.x, self.y, self.w, self.h]);
        for (i, landmark) in self.landmarks.iter().enumerate() {
            row[4 + i * 2..6 + i * 2].copy_from_slice(landmark);
        }
        row[14] = self.score;
        row
    }
}
pub struct FaceDetector {
    ctx: *mut ffi::FaceCtx,
}
// The native context is accessed exclusively through &mut self / the provider mutex.
unsafe impl Send for FaceDetector {}
impl FaceDetector {
    pub fn load() -> Result<Self> {
        Self::create(false)
    }
    pub(super) fn load_with_rec() -> Result<Self> {
        Self::create(true)
    }
    fn create(recognize: bool) -> Result<Self> {
        let detector = CString::new(
            ensure_model(&DET_MODEL)?
                .to_str()
                .context("model path is not UTF-8")?,
        )?;
        let recognizer = if recognize {
            Some(CString::new(
                ensure_model(&REC_MODEL)?
                    .to_str()
                    .context("model path is not UTF-8")?,
            )?)
        } else {
            None
        };
        let ctx = unsafe {
            ffi::face_create(
                detector.as_ptr(),
                recognizer.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()),
            )
        };
        if ctx.is_null() {
            bail!("YuNet/SFace initialization failed: {}", last_error());
        }
        Ok(Self { ctx })
    }
    pub(super) fn raw_ctx(&mut self) -> *mut ffi::FaceCtx {
        self.ctx
    }
    pub fn detect(&mut self, frame: &GrayscaleFrame) -> Result<Vec<Detection>> {
        frame.validate()?;
        detect_on_pixels(self.ctx, &frame.pixels, frame.width, frame.height, 1)
    }
}
impl Drop for FaceDetector {
    fn drop(&mut self) {
        unsafe { ffi::face_destroy(self.ctx) }
    }
}
pub(super) fn detect_on_ctx(ctx: *mut ffi::FaceCtx, frame: &BgrFrame) -> Result<Vec<Detection>> {
    frame.validate()?;
    detect_on_pixels(ctx, &frame.pixels, frame.width, frame.height, 3)
}
fn detect_on_pixels(
    ctx: *mut ffi::FaceCtx,
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: i32,
) -> Result<Vec<Detection>> {
    const MAX_FACES: usize = 16;
    let mut rows = [0.0f32; MAX_FACES * 15];
    let count = unsafe {
        ffi::face_detect(
            ctx,
            pixels.as_ptr(),
            width as i32,
            height as i32,
            channels,
            rows.as_mut_ptr(),
            MAX_FACES as i32,
        )
    };
    if count < 0 || count as usize > MAX_FACES {
        bail!("YuNet detection failed: {}", last_error());
    }
    let mut detections = Vec::new();
    for row in rows.as_chunks::<15>().0.iter().take(count as usize) {
        anyhow::ensure!(
            row.iter().all(|x| x.is_finite()) && row[2] > 0.0 && row[3] > 0.0,
            "YuNet returned an invalid detection"
        );
        detections.push(Detection {
            x: row[0],
            y: row[1],
            w: row[2],
            h: row[3],
            landmarks: [
                [row[4], row[5]],
                [row[6], row[7]],
                [row[8], row[9]],
                [row[10], row[11]],
                [row[12], row[13]],
            ],
            score: row[14],
        });
    }
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(detections)
}
pub(super) fn last_error() -> String {
    unsafe {
        std::ffi::CStr::from_ptr(ffi::face_last_error())
            .to_string_lossy()
            .into_owned()
    }
}
