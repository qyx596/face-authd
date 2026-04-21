pub mod detector;
pub mod embedder;
pub mod enrollment;
pub mod models;
mod ffi;

use anyhow::{Context, Result};

use crate::camera::GrayscaleFrame;
use detector::{detect_on_ctx, load_models, FaceDetector};
use embedder::{embed_on_ctx, cosine_similarity, Embedding};

/// Cosine similarity threshold for a positive match.
/// dlib's ResNet descriptors are L2-normalised; howdy-equivalent default is ~0.6.
pub const DEFAULT_THRESHOLD: f32 = 0.6;

pub struct FaceRecognizer {
    detector: FaceDetector,
}

impl FaceRecognizer {
    pub fn load() -> Result<Self> {
        let (sp_path, rec_path) = load_models().context("failed to load dlib models")?;
        let detector = FaceDetector::load_with_rec(&sp_path, &rec_path)
            .context("failed to initialise dlib context")?;
        Ok(Self { detector })
    }

    /// Detect the best face and return its embedding, or None if no face found.
    pub fn extract(&mut self, frame: &GrayscaleFrame) -> Result<Option<Embedding>> {
        let ctx = self.detector.raw_ctx();
        let detections = detect_on_ctx(ctx, frame).context("face detection failed")?;

        let Some(best) = detections.into_iter().next() else {
            return Ok(None);
        };

        let embedding = embed_on_ctx(ctx, frame, &best).context("face embedding failed")?;
        Ok(Some(embedding))
    }

    /// Return the maximum cosine similarity between a query and all enrolled embeddings.
    pub fn match_score(query: &Embedding, enrolled: &[Embedding]) -> f32 {
        enrolled
            .iter()
            .map(|e| cosine_similarity(query, e))
            .fold(f32::NEG_INFINITY, f32::max)
    }
}
