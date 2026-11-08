use anyhow::{bail, Result};

use super::detector::{last_error, Detection};
use super::ffi;
use crate::camera::BgrFrame;

/// 128-dimensional, explicitly L2-normalized SFace descriptor.
pub type Embedding = [f32; 128];

pub(super) fn embed_on_ctx(
    ctx: *mut ffi::FaceCtx,
    frame: &BgrFrame,
    det: &Detection,
) -> Result<Embedding> {
    frame.validate()?;
    let mut emb = [0f32; 128];

    let ret = unsafe {
        ffi::face_embed(
            ctx,
            frame.pixels.as_ptr(),
            frame.width as i32,
            frame.height as i32,
            3,
            det.row().as_ptr(),
            emb.as_mut_ptr(),
        )
    };

    if ret != 0 {
        bail!("SFace extraction failed: {}", last_error());
    }

    if !valid_embedding(&emb) {
        bail!("SFace returned an invalid face descriptor");
    }
    Ok(emb)
}

/// Euclidean distance between unit SFace descriptors. Lower is better.
/// Invalid descriptors must never produce a matching distance.
pub fn euclidean_distance(a: &Embedding, b: &Embedding) -> f32 {
    if !valid_embedding(a) || !valid_embedding(b) {
        return f32::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

pub fn valid_embedding(a: &Embedding) -> bool {
    a.iter().all(|x| x.is_finite()) && (a.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() <= 0.001
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_feature_distances_and_scale_validation() {
        let mut a = [0.0; 128];
        a[0] = 1.0;
        let mut b = [0.0; 128];
        b[1] = 1.0;
        assert_eq!(euclidean_distance(&a, &a), 0.0);
        assert!((euclidean_distance(&a, &b) - 2.0_f32.sqrt()).abs() < 1e-6);
        b[1] = 2.0;
        assert_eq!(euclidean_distance(&a, &b), f32::INFINITY);
    }

    #[test]
    fn invalid_descriptors_fail_closed() {
        let a = [1.0_f32 / 128.0_f32.sqrt(); 128];
        for bad in [[0.0; 128], [f32::NAN; 128], [f32::INFINITY; 128]] {
            assert_eq!(euclidean_distance(&a, &bad), f32::INFINITY);
        }
    }
}
