pub mod detector;
pub mod embedder;
pub mod enrollment;
mod ffi;
pub mod models;

use anyhow::{Context, Result};

use crate::camera::BgrFrame;
use detector::{detect_on_ctx, FaceDetector};
use embedder::{embed_on_ctx, euclidean_distance, Embedding};

/// Maximum Euclidean distance between normalized SFace features (strictly less).
/// This is a starting point, not a calibrated IR false-accept guarantee.
pub const DEFAULT_THRESHOLD: f32 = 1.128;

pub struct FaceObservation {
    pub faces: usize,
    pub embedding: Option<Embedding>,
}

pub struct FaceRecognizer {
    detector: FaceDetector,
}

impl FaceRecognizer {
    pub fn load() -> Result<Self> {
        let detector = FaceDetector::load_with_rec().context("failed to initialize YuNet/SFace")?;
        Ok(Self { detector })
    }

    /// Return a descriptor only when exactly one face is present.
    pub fn extract(&mut self, frame: &BgrFrame) -> Result<Option<Embedding>> {
        Ok(self.analyze(frame)?.embedding)
    }

    pub fn analyze(&mut self, frame: &BgrFrame) -> Result<FaceObservation> {
        let ctx = self.detector.raw_ctx();
        let detections = detect_on_ctx(ctx, frame).context("YuNet detection failed")?;
        let faces = detections.len();
        let embedding = if faces == 1 {
            Some(embed_on_ctx(ctx, frame, &detections[0]).context("SFace embedding failed")?)
        } else {
            None
        };
        Ok(FaceObservation { faces, embedding })
    }

    /// Return the minimum Euclidean distance between a query and all enrolled embeddings.
    pub fn match_distance(query: &Embedding, enrolled: &[Embedding]) -> f32 {
        enrolled
            .iter()
            .map(|e| euclidean_distance(query, e))
            .fold(f32::INFINITY, f32::min)
    }
}

pub fn validate_threshold(threshold: f32) -> Result<()> {
    anyhow::ensure!(
        threshold.is_finite() && threshold > 0.0 && threshold <= DEFAULT_THRESHOLD,
        "distance threshold must be finite and in (0, 1.128]; lower is stricter"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_enrollment_never_matches() {
        assert_eq!(
            FaceRecognizer::match_distance(&[1.0; 128], &[]),
            f32::INFINITY
        );
    }
    #[test]
    fn reject_unsafe_thresholds() {
        for t in [f32::NAN, f32::INFINITY, 0.0, -0.1, 1.129] {
            assert!(validate_threshold(t).is_err());
        }
        assert!(validate_threshold(DEFAULT_THRESHOLD).is_ok());
    }
}

#[cfg(test)]
mod native_tests {
    use super::*;
    #[test]
    #[ignore = "requires verified model files via FACE_AUTHD_MODEL_DIR"]
    fn real_models_load_and_blank_frame_has_no_face() {
        let mut r = FaceRecognizer::load().unwrap();
        let frame = BgrFrame {
            width: 640,
            height: 360,
            pixels: vec![0; 640 * 360 * 3],
        };
        assert!(r.extract(&frame).unwrap().is_none());
        // Exercise SFace alignment/inference as well as YuNet on synthetic data.
        let detection = detector::Detection {
            x: 100.0,
            y: 70.0,
            w: 160.0,
            h: 180.0,
            landmarks: [
                [145.0, 130.0],
                [215.0, 130.0],
                [180.0, 165.0],
                [155.0, 205.0],
                [205.0, 205.0],
            ],
            score: 1.0,
        };
        let feature = embedder::embed_on_ctx(r.detector.raw_ctx(), &frame, &detection).unwrap();
        assert!(embedder::valid_embedding(&feature));
        assert_eq!(FaceRecognizer::match_distance(&feature, &[feature]), 0.0);

        let bad = BgrFrame {
            width: 640,
            height: 360,
            pixels: vec![0; 10],
        };
        assert!(r.extract(&bad).is_err());
    }
}

/// Shared decision rule for diagnostic verification and PAM authentication.
pub struct MatchTracker {
    threshold: f32,
    consecutive: usize,
}
impl MatchTracker {
    pub fn new(threshold: f32) -> Result<Self> {
        validate_threshold(threshold)?;
        Ok(Self {
            threshold,
            consecutive: 0,
        })
    }
    pub fn observe(&mut self, distance: f32) -> bool {
        self.consecutive = if distance.is_finite() && distance >= 0.0 && distance < self.threshold {
            self.consecutive + 1
        } else {
            0
        };
        self.consecutive >= 2
    }
}

#[cfg(test)]
mod match_tests {
    use super::*;
    #[test]
    fn require_adjacent_matches_and_strict_boundary() {
        let mut tracker = MatchTracker::new(0.6).unwrap();
        assert!(!tracker.observe(0.2));
        assert!(!tracker.observe(0.6));
        assert!(!tracker.observe(0.2));
        assert!(tracker.observe(0.3));
        assert!(!tracker.observe(f32::NAN));
        assert!(!tracker.observe(0.3));
        assert!(!tracker.observe(f32::INFINITY));
        assert!(!tracker.observe(-1.0));
    }
}
