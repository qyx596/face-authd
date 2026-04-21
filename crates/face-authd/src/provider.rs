use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use common::protocol::AuthenticateRequest;

use crate::camera::{
    decode_to_grayscale, discover_video_devices, preferred_device_from_env, select_capture_device,
    stream_frames, CaptureConfig, StreamControl,
};
use crate::emitter::{emitter_from_env, EmitterController};
use crate::recognition::{
    enrollment::load_enrollment, FaceRecognizer,
};
use tracing::info;

pub type ProviderFuture<'a> = Pin<Box<dyn Future<Output = AuthDecision> + Send + 'a>>;

pub enum AuthDecision {
    Allow { reason: Option<String> },
    Deny { reason: Option<String> },
    Error { message: String },
}

pub trait FaceAuthProvider: Send + Sync {
    fn authenticate<'a>(&'a self, request: &'a AuthenticateRequest) -> ProviderFuture<'a>;
}

// ---------------------------------------------------------------------------
// Real provider
// ---------------------------------------------------------------------------

pub struct FaceProvider {
    recognizer: Arc<Mutex<FaceRecognizer>>,
    emitter: Arc<dyn EmitterController>,
    /// Minimum cosine similarity to accept a match.
    threshold: f32,
}

impl FaceProvider {
    pub fn new(recognizer: FaceRecognizer, threshold: f32) -> Self {
        Self {
            recognizer: Arc::new(Mutex::new(recognizer)),
            emitter: Arc::new(emitter_from_env()),
            threshold,
        }
    }
}

impl FaceAuthProvider for FaceProvider {
    fn authenticate<'a>(&'a self, request: &'a AuthenticateRequest) -> ProviderFuture<'a> {
        Box::pin(async move {
            // Load enrollment synchronously (fast file I/O)
            let enrollment = match load_enrollment(&request.username) {
                Ok(Some(e)) => e,
                Ok(None) => {
                    return AuthDecision::Deny {
                        reason: Some(format!(
                            "no face enrollment for user '{}'",
                            request.username
                        )),
                    };
                }
                Err(err) => {
                    return AuthDecision::Error {
                        message: format!("failed to load enrollment: {err}"),
                    };
                }
            };

            let enrolled = enrollment.embeddings_as_arrays();
            if enrolled.is_empty() {
                return AuthDecision::Deny {
                    reason: Some("enrollment exists but contains no valid embeddings".to_string()),
                };
            }

            // Select camera
            let devices = match discover_video_devices() {
                Ok(d) => d,
                Err(err) => {
                    return AuthDecision::Error {
                        message: format!("camera discovery failed: {err}"),
                    };
                }
            };

            let preferred = preferred_device_from_env();
            let fallback_device = select_capture_device(&devices, preferred.as_deref());
            let ir_device = match devices
                .iter()
                .find(|d| {
                    d.supports_capture
                        && d.supports_streaming
                        && matches!(d.fourcc.as_str(), "GREY" | "Y8")
                })
                .or_else(|| fallback_device.as_ref())
            {
                Some(d) => d.path.clone(),
                None => {
                    return AuthDecision::Error {
                        message: "no suitable IR camera found".to_string(),
                    };
                }
            };

            // Activate IR emitter
            if let Err(err) = self.emitter.activate(Some(&ir_device)) {
                return AuthDecision::Error {
                    message: format!("emitter activation failed: {err}"),
                };
            }

            // Capture frames and try to get a face embedding
            // Try up to 5 frames, use the best match score
            let cap_config = CaptureConfig::default();
            let recognizer = Arc::clone(&self.recognizer);
            let threshold = self.threshold;

            // Run blocking camera + inference on a thread pool thread
            let result = tokio::task::spawn_blocking(move || {
                let mut best_score = f32::NEG_INFINITY;
                let mut face_found = false;
                let mut processed = 0usize;

                stream_frames(&ir_device, &cap_config, 2, |frame| {
                    if processed >= 5 {
                        return Ok(StreamControl::Stop);
                    }
                    processed += 1;
                    let gray = decode_to_grayscale(&frame)?;

                    let mut recognizer = recognizer
                        .lock()
                        .map_err(|_| anyhow::anyhow!("recognizer mutex poisoned"))?;

                    if let Some(embedding) = recognizer.extract(&gray)? {
                        face_found = true;
                        let score = FaceRecognizer::match_score(&embedding, &enrolled);
                        if score > best_score {
                            best_score = score;
                        }
                        if best_score >= threshold {
                            return Ok(StreamControl::Stop);
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    Ok(StreamControl::Continue)
                })?;

                Ok::<_, anyhow::Error>((face_found, best_score))
            })
            .await;

            match result {
                Ok(Ok((face_found, score))) => {
                    if !face_found {
                        info!(username = %request.username, "auth: no face detected");
                        AuthDecision::Deny {
                            reason: Some("no face detected in camera frame".to_string()),
                        }
                    } else if score >= threshold {
                        info!(
                            username = %request.username,
                            score = format!("{score:.4}"),
                            threshold = format!("{threshold:.4}"),
                            "auth: ALLOW"
                        );
                        AuthDecision::Allow {
                            reason: Some(format!(
                                "face matched (score={score:.4}, threshold={threshold:.4})"
                            )),
                        }
                    } else {
                        info!(
                            username = %request.username,
                            score = format!("{score:.4}"),
                            threshold = format!("{threshold:.4}"),
                            "auth: DENY — score below threshold"
                        );
                        AuthDecision::Deny {
                            reason: Some(format!(
                                "face not recognized (score={score:.4}, threshold={threshold:.4})"
                            )),
                        }
                    }
                }
                Ok(Err(err)) => AuthDecision::Error {
                    message: format!("recognition failed: {err}"),
                },
                Err(err) => AuthDecision::Error {
                    message: format!("task panicked: {err}"),
                },
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Stub provider (kept for testing without models)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct StubProvider;

impl FaceAuthProvider for StubProvider {
    fn authenticate<'a>(&'a self, request: &'a AuthenticateRequest) -> ProviderFuture<'a> {
        Box::pin(async move {
            if request.username.is_empty() {
                return AuthDecision::Error {
                    message: "username must not be empty".to_string(),
                };
            }
            if matches!(
                std::env::var("FACE_AUTHD_STUB_ALLOW").ok().as_deref(),
                Some("1" | "true" | "yes")
            ) {
                return AuthDecision::Allow {
                    reason: Some("stub provider: FACE_AUTHD_STUB_ALLOW is set".to_string()),
                };
            }
            AuthDecision::Deny {
                reason: Some("stub provider in use; enroll first and restart daemon".to_string()),
            }
        })
    }
}
