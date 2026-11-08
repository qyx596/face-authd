//! Live measurements use in-memory references; images and embeddings are never saved.
use crate::camera::{self, CaptureConfig, StreamControl};
use crate::emitter::{emitter_from_env, EmitterController};
use crate::recognition::{FaceRecognizer, MatchTracker, DEFAULT_THRESHOLD};
use anyhow::{Context, Result};
use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Default, Serialize)]
struct Phase {
    frames: usize,
    single_face_frames: usize,
    no_face_frames: usize,
    multiple_face_frames: usize,
    matching_frames: usize,
    confirmed_match: bool,
    first_confirmed_match_ms: Option<f64>,
    inference_p50_ms: f64,
    inference_p95_ms: f64,
    decode_p50_ms: f64,
    mean_pixel_value: f64,
    frame_width: u32,
    frame_height: u32,
    capture_fourcc: String,
    wall_seconds: f64,
    #[serde(skip)]
    inference: Vec<f64>,
    #[serde(skip)]
    decode: Vec<f64>,
}
#[derive(Serialize)]
struct Report {
    model_id: &'static str,
    device: String,
    threshold_l2: f32,
    enrollment_samples: usize,
    enrollment: Phase,
    verification: Phase,
    minimum_verification_distance: Option<f32>,
    limitations: &'static str,
}
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * p).ceil() as usize]
}
impl Phase {
    fn finish(&mut self, start: Instant) {
        self.wall_seconds = start.elapsed().as_secs_f64();
        self.inference_p50_ms = percentile(&self.inference, 0.5);
        self.inference_p95_ms = percentile(&self.inference, 0.95);
        self.decode_p50_ms = percentile(&self.decode, 0.5);
        if self.frames > 0 {
            self.mean_pixel_value /= self.frames as f64;
        }
    }
}
pub fn run(args: &[String]) -> Result<()> {
    let mut device: Option<PathBuf> = None;
    let mut frames = 40usize;
    let mut countdown = 5u64;
    let mut output: Option<PathBuf> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--device" => device = Some(iter.next().context("--device requires a path")?.into()),
            "--frames" => frames = iter.next().context("--frames requires a count")?.parse()?,
            "--countdown" => {
                countdown = iter
                    .next()
                    .context("--countdown requires seconds")?
                    .parse()?
            }
            "--json" => output = Some(iter.next().context("--json requires a path")?.into()),
            other => anyhow::bail!("unknown benchmark argument: {other}"),
        }
    }
    anyhow::ensure!(
        (2..=300).contains(&frames) && countdown <= 30,
        "frames must be 2..=300; countdown 0..=30"
    );
    let devices = camera::discover_video_devices()?;
    let path = match device {
        Some(p) => p,
        None => camera::select_auth_device(&devices)?.path,
    };
    let canonical = std::fs::canonicalize(&path)?;
    let info = devices
        .iter()
        .find(|d| d.path == canonical)
        .context("camera is not a streaming capture node")?;
    let config = CaptureConfig {
        width: Some(640),
        height: Some(360),
        fourcc: Some(if matches!(info.fourcc.as_str(), "GREY" | "Y8") {
            *b"GREY"
        } else {
            *b"MJPG"
        }),
        ..Default::default()
    };
    eprintln!("Loading YuNet/SFace; live test camera {}", path.display());
    let mut recognizer = FaceRecognizer::load()?;
    eprintln!("Look at the camera: sampling starts in {countdown}s. No frames or descriptors are written to disk.");
    std::thread::sleep(Duration::from_secs(countdown));
    emitter_from_env().activate(Some(&path))?;
    let mut references = Vec::new();
    let mut enrollment = Phase::default();
    let start = Instant::now();
    camera::stream_frames(&path, &config, 5, |frame| {
        let decode_start = Instant::now();
        let bgr = camera::decode_to_bgr(&frame)?;
        enrollment
            .decode
            .push(decode_start.elapsed().as_secs_f64() * 1000.0);
        let inference_start = Instant::now();
        let observation = recognizer.analyze(&bgr)?;
        enrollment
            .inference
            .push(inference_start.elapsed().as_secs_f64() * 1000.0);
        record(&mut enrollment, &frame, &bgr, observation.faces);
        if let Some(embedding) = observation.embedding {
            references.push(embedding);
        }
        if references.len() >= 5 || enrollment.frames >= frames {
            return Ok(StreamControl::Stop);
        }
        std::thread::sleep(Duration::from_millis(150));
        Ok(StreamControl::Continue)
    })?;
    enrollment.finish(start);
    eprintln!("Reference samples: {}. Release/reopen camera; verification in 3s. Keep looking at the camera.", references.len());
    std::thread::sleep(Duration::from_secs(3));
    emitter_from_env().activate(Some(&path))?;
    let mut verification = Phase::default();
    let mut tracker = MatchTracker::new(DEFAULT_THRESHOLD)?;
    let mut minimum = f32::INFINITY;
    let start = Instant::now();
    camera::stream_frames(&path, &config, 5, |frame| {
        let decode_start = Instant::now();
        let bgr = camera::decode_to_bgr(&frame)?;
        verification
            .decode
            .push(decode_start.elapsed().as_secs_f64() * 1000.0);
        let inference_start = Instant::now();
        let observation = recognizer.analyze(&bgr)?;
        verification
            .inference
            .push(inference_start.elapsed().as_secs_f64() * 1000.0);
        record(&mut verification, &frame, &bgr, observation.faces);
        let distance = observation.embedding.as_ref().map_or(f32::INFINITY, |e| {
            FaceRecognizer::match_distance(e, &references)
        });
        minimum = minimum.min(distance);
        verification.matching_frames += usize::from(distance < DEFAULT_THRESHOLD);
        if tracker.observe(distance) {
            verification.confirmed_match = true;
            verification
                .first_confirmed_match_ms
                .get_or_insert(start.elapsed().as_secs_f64() * 1000.0);
        }
        if verification.frames >= frames {
            return Ok(StreamControl::Stop);
        }
        std::thread::sleep(Duration::from_millis(100));
        Ok(StreamControl::Continue)
    })?;
    verification.finish(start);
    let report = Report { model_id: crate::recognition::enrollment::MODEL_ID, device: path.display().to_string(),
        threshold_l2: DEFAULT_THRESHOLD, enrollment_samples: references.len(), enrollment, verification,
        minimum_verification_distance: minimum.is_finite().then_some(minimum),
        limitations: "Short same-session smoke test with a camera reopen; no identity proof, independent-session accuracy, impostor rate, spoof resistance or PAM integration measurement. References are memory-only." };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        std::fs::write(path, &json)?;
    }
    println!("{json}");
    Ok(())
}
fn record(phase: &mut Phase, frame: &camera::CapturedFrame, bgr: &camera::BgrFrame, faces: usize) {
    phase.frames += 1;
    match faces {
        0 => phase.no_face_frames += 1,
        1 => phase.single_face_frames += 1,
        _ => phase.multiple_face_frames += 1,
    }
    phase.mean_pixel_value +=
        bgr.pixels.iter().map(|v| u64::from(*v)).sum::<u64>() as f64 / bgr.pixels.len() as f64;
    phase.frame_width = bgr.width;
    phase.frame_height = bgr.height;
    phase.capture_fourcc = frame.fourcc.clone();
}
#[cfg(test)]
mod tests {
    #[test]
    fn percentiles_are_order_independent() {
        assert_eq!(super::percentile(&[4.0, 1.0, 3.0, 2.0], 0.95), 4.0);
        assert_eq!(super::percentile(&[], 0.5), 0.0);
    }
}
