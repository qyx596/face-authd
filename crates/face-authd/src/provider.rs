use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::camera::{
    decode_to_bgr, discover_video_devices, select_auth_device, stream_frames, StreamControl,
};
use crate::emitter::{emitter_from_env, EmitterController};
use crate::recognition::{enrollment::load_enrollment, FaceRecognizer};
use common::protocol::AuthenticateRequest;
use tokio::sync::Semaphore;

pub type ProviderFuture<'a> = Pin<Box<dyn Future<Output = AuthDecision> + Send + 'a>>;
pub enum AuthDecision {
    Allow { reason: Option<String> },
    Deny { reason: Option<String> },
    Error { message: String },
}
pub trait FaceAuthProvider: Send + Sync {
    fn authenticate<'a>(&'a self, request: &'a AuthenticateRequest) -> ProviderFuture<'a>;
}

pub struct FaceProvider {
    recognizer: Arc<Mutex<FaceRecognizer>>,
    emitter: Arc<dyn EmitterController>,
    threshold: f32,
    camera: Arc<Semaphore>,
}

// Dropping an async request also signals its blocking worker. The worker retains
// the camera permit until it has actually released the device.
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

impl FaceProvider {
    pub fn new(recognizer: FaceRecognizer, threshold: f32) -> Self {
        Self {
            recognizer: Arc::new(Mutex::new(recognizer)),
            emitter: Arc::new(emitter_from_env()),
            threshold,
            camera: Arc::new(Semaphore::new(1)),
        }
    }
}

impl FaceAuthProvider for FaceProvider {
    fn authenticate<'a>(&'a self, request: &'a AuthenticateRequest) -> ProviderFuture<'a> {
        Box::pin(async move {
            let Ok(permit) = Arc::clone(&self.camera).try_acquire_owned() else {
                return AuthDecision::Error {
                    message: "camera busy; use password authentication".into(),
                };
            };
            let cancelled = Arc::new(AtomicBool::new(false));
            let _cancel_guard = CancelOnDrop(Arc::clone(&cancelled));
            let recognizer = Arc::clone(&self.recognizer);
            let emitter = Arc::clone(&self.emitter);
            let username = request.username.clone();
            let threshold = self.threshold;
            let deadline = Instant::now() + Duration::from_secs(8);
            let job = tokio::task::spawn_blocking(move || -> anyhow::Result<bool> {
                let _permit = permit;
                let enrollment = load_enrollment(&username)?;
                let Some(enrollment) = enrollment else {
                    return Ok(false);
                };
                let enrolled = enrollment.embeddings_as_arrays();
                if enrolled.is_empty() {
                    return Ok(false);
                }
                let devices = discover_video_devices()?;
                let device = select_auth_device(&devices)?;
                emitter.activate(Some(&device.path))?;
                let mut recognizer = recognizer
                    .lock()
                    .map_err(|_| anyhow::anyhow!("recognizer mutex poisoned"))?;
                let capture_deadline = Instant::now() + Duration::from_secs(4);
                let mut tracker = crate::recognition::MatchTracker::new(threshold)?;
                let mut matched = false;
                stream_frames(
                    &device.path,
                    &crate::camera::auth_capture_config(),
                    5,
                    |frame| {
                        anyhow::ensure!(
                            !cancelled.load(Ordering::Acquire) && Instant::now() < deadline,
                            "authentication cancelled or timed out"
                        );
                        let gray = decode_to_bgr(&frame)?;
                        let distance = match recognizer.extract(&gray)? {
                            Some(embedding) => {
                                FaceRecognizer::match_distance(&embedding, &enrolled)
                            }
                            None => f32::INFINITY,
                        };
                        // This confirms a match across frames; it is not liveness detection.
                        matched = tracker.observe(distance);
                        Ok(if matched || Instant::now() >= capture_deadline {
                            StreamControl::Stop
                        } else {
                            StreamControl::Continue
                        })
                    },
                )?;
                anyhow::ensure!(
                    !cancelled.load(Ordering::Acquire) && Instant::now() < deadline,
                    "authentication cancelled or timed out"
                );
                Ok(matched)
            });
            match tokio::time::timeout(Duration::from_secs(8), job).await {
                Ok(Ok(Ok(true))) => AuthDecision::Allow {
                    reason: Some("face matched on consecutive frames".into()),
                },
                Ok(Ok(Ok(false))) => AuthDecision::Deny {
                    reason: Some("no consistent face match".into()),
                },
                Ok(Ok(Err(err))) => AuthDecision::Error {
                    message: format!("recognition failed: {err}"),
                },
                Ok(Err(err)) => AuthDecision::Error {
                    message: format!("recognition worker failed: {err}"),
                },
                Err(_) => AuthDecision::Error {
                    message: "authentication timed out; use password authentication".into(),
                },
            }
        })
    }
}
