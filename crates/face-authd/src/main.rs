mod camera;
mod emitter;
mod provider;
mod raw_controls;
mod recognition;
mod vision;

use std::env;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::Context;
use camera::{
    capture_frames, capture_single_frame, decode_to_grayscale, discover_video_devices,
    preferred_device_from_env, select_capture_device, write_frame_to_path,
    stream_frames, write_grayscale_pgm, CaptureConfig, StreamControl,
};
use common::protocol::{
    encode_response, AuthenticateRequest, AuthenticateResponse, ErrorResponse, Request, Response,
    DEFAULT_SOCKET_PATH, PROTOCOL_VERSION,
};
use eframe::egui;
use provider::{AuthDecision, FaceAuthProvider, FaceProvider, StubProvider};
use raw_controls::{query_device_controls, set_device_control};
use recognition::{
    detector::{Detection, FaceDetector},
    enrollment::{
        check_enrollment_dir, delete_enrollment, enrollment_dir, list_enrolled_users,
        load_enrollment, save_enrollment, UserEnrollment,
    },
    FaceRecognizer, DEFAULT_THRESHOLD,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{error, info, warn};
use vision::{format_analysis, FrameAnalyzer, StubFrameAnalyzer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,face_authd=debug".to_string()),
        )
        .init();

    let args: Vec<String> = env::args().collect();
    if matches!(args.get(1).map(String::as_str), Some("probe-cameras")) {
        return run_probe_cameras(&args[2..]);
    }
    if matches!(args.get(1).map(String::as_str), Some("hardware-survey")) {
        return run_hardware_survey();
    }
    if matches!(args.get(1).map(String::as_str), Some("enroll")) {
        return run_enroll(&args[2..]);
    }
    if matches!(args.get(1).map(String::as_str), Some("setup")) {
        return run_setup(&args[2..]);
    }
    if matches!(args.get(1).map(String::as_str), Some("enroll-verify" | "onboard")) {
        return run_enroll_verify(&args[2..]);
    }
    if matches!(args.get(1).map(String::as_str), Some("enrolled")) {
        return run_enrolled();
    }
    if matches!(args.get(1).map(String::as_str), Some("unenroll")) {
        return run_unenroll(&args[2..]);
    }
    if matches!(args.get(1).map(String::as_str), Some("preview")) {
        return run_preview(&args[2..]);
    }
    if matches!(args.get(1).map(String::as_str), Some("verify")) {
        return run_verify(&args[2..]);
    }

    if matches!(
        args.get(1).map(String::as_str),
        Some("-h" | "--help" | "help")
    ) {
        print_usage();
        return Ok(());
    }

    run_daemon().await
}

async fn run_daemon() -> anyhow::Result<()> {
    let socket_path =
        std::env::var("FACE_AUTHD_SOCKET").unwrap_or_else(|_| DEFAULT_SOCKET_PATH.to_string());

    if Path::new(&socket_path).exists() {
        std::fs::remove_file(&socket_path)
            .with_context(|| format!("failed to remove stale socket at {socket_path}"))?;
    }

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("failed to bind unix socket at {socket_path}"))?;

    info!("face-authd listening on {}", socket_path);

    let provider: Arc<dyn FaceAuthProvider> = match load_face_recognizer_blocking() {
        Ok(recognizer) => {
            info!("face recognition models loaded");
            Arc::new(FaceProvider::new(recognizer, DEFAULT_THRESHOLD))
        }
        Err(err) => {
            warn!(error = %err, "failed to load recognition models; falling back to stub provider");
            Arc::new(StubProvider::default())
        }
    };

    loop {
        let (stream, _) = listener.accept().await?;
        let provider = Arc::clone(&provider);

        tokio::spawn(async move {
            if let Err(err) = handle_client(stream, provider).await {
                error!(error = %err, "client handling failed");
            }
        });
    }
}

fn load_face_recognizer_blocking() -> anyhow::Result<FaceRecognizer> {
    tokio::task::block_in_place(FaceRecognizer::load)
}

fn load_face_detector_blocking() -> anyhow::Result<FaceDetector> {
    tokio::task::block_in_place(FaceDetector::load)
}

fn run_probe_cameras(args: &[String]) -> anyhow::Result<()> {
    let mut do_capture = false;
    let mut preferred = preferred_device_from_env();
    let mut width = None;
    let mut height = None;
    let mut fourcc = None;
    let mut output: Option<PathBuf> = None;
    let mut output_gray: Option<PathBuf> = None;
    let mut analyze_gray = false;
    let mut burst_count: Option<usize> = None;
    let mut list_controls = false;
    let mut set_controls = Vec::new();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--capture" => do_capture = true,
            "--device" => {
                let value = iter.next().context(
                    "missing value for --device; expected a /dev/video* or /dev/v4l/by-path path",
                )?;
                preferred = Some(value.into());
            }
            "--width" => {
                let value = iter
                    .next()
                    .context("missing value for --width; expected an integer")?;
                width = Some(value.parse().context("invalid integer for --width")?);
            }
            "--height" => {
                let value = iter
                    .next()
                    .context("missing value for --height; expected an integer")?;
                height = Some(value.parse().context("invalid integer for --height")?);
            }
            "--fourcc" => {
                let value = iter.next().context(
                    "missing value for --fourcc; expected a 4-character code such as YUYV or MJPG",
                )?;
                fourcc = Some(parse_fourcc(value)?);
            }
            "--output" => {
                let value = iter
                    .next()
                    .context("missing value for --output; expected a filesystem path")?;
                output = Some(value.into());
            }
            "--output-gray" => {
                let value = iter
                    .next()
                    .context("missing value for --output-gray; expected a filesystem path")?;
                output_gray = Some(value.into());
            }
            "--analyze-gray" => analyze_gray = true,
            "--burst" => {
                let value = iter
                    .next()
                    .context("missing value for --burst; expected a positive integer")?;
                burst_count = Some(value.parse().context("invalid integer for --burst")?);
            }
            "--list-controls" => list_controls = true,
            "--set-control" => {
                let value = iter
                    .next()
                    .context("missing value for --set-control; expected name=value")?;
                set_controls.push(value.to_string());
            }
            "-h" | "--help" => {
                print_probe_usage();
                return Ok(());
            }
            other => anyhow::bail!("unknown probe-cameras argument: {other}"),
        }
    }

    let devices = discover_video_devices()?;
    if devices.is_empty() {
        println!("No V4L2 video devices found.");
        return Ok(());
    }

    let selected = select_capture_device(&devices, preferred.as_deref());

    println!("Discovered {} video device(s):", devices.len());
    for device in &devices {
        let marker = if selected
            .as_ref()
            .map(|selected| selected.path == device.path)
            .unwrap_or(false)
        {
            "*"
        } else {
            " "
        };

        println!(
            "{} {}",
            marker,
            format_device_line(device, device.symlinks.first().map(|path| path.as_path()))
        );

        if device.symlinks.len() > 1 {
            for link in device.symlinks.iter().skip(1) {
                println!("    by-path: {}", link.display());
            }
        }
    }

    if list_controls || !set_controls.is_empty() {
        let Some(device) = selected.as_ref() else {
            anyhow::bail!("no capture-capable device available for control inspection");
        };

        for spec in &set_controls {
            let (name, value) = spec.split_once('=').with_context(|| {
                format!(
                    "invalid --set-control value '{}'; expected name=value",
                    spec
                )
            })?;
            set_device_control(&device.path, name.trim(), value.trim())?;
            println!("Set control '{}' to '{}'", name.trim(), value.trim());
        }

        if list_controls {
            let controls = query_device_controls(&device.path)?;
            println!();
            println!("Controls for {}:", device.path.display());
            for control in controls {
                println!(
                    "  id={} name=\"{}\" type={} current={} range=[{}..{}] step={} default={} flags={}",
                    control.id,
                    control.name,
                    control.typ_name,
                    control.current.unwrap_or_else(|| "-".to_string()),
                    control.minimum,
                    control.maximum,
                    control.step,
                    control.default,
                    control.flags_text
                );
            }
        }
    }

    if do_capture {
        let analyzer = StubFrameAnalyzer;
        let Some(device) = selected else {
            anyhow::bail!("no capture-capable device available for --capture");
        };

        let frame = capture_single_frame(
            &device.path,
            &CaptureConfig {
                width,
                height,
                fourcc,
                ..CaptureConfig::default()
            },
        )?;

        println!();
        println!(
            "Capture probe succeeded: device={} frame={}x{} fourcc={} seq={} bytes={}",
            frame.device.display(),
            frame.width,
            frame.height,
            frame.fourcc,
            frame.sequence,
            frame.bytes.len()
        );

        if let Some(output) = output.as_deref() {
            let file_kind = write_frame_to_path(&frame, output)?;
            println!("Saved {file_kind} frame to {}", output.display());
        }

        if let Some(output_gray) = output_gray.as_deref() {
            let grayscale = decode_to_grayscale(&frame)?;
            write_grayscale_pgm(&grayscale, output_gray)?;
            println!(
                "Saved grayscale frame to {} ({}x{})",
                output_gray.display(),
                grayscale.width,
                grayscale.height
            );
        }

        if analyze_gray {
            let grayscale = decode_to_grayscale(&frame)?;
            let analysis = analyzer.analyze(&grayscale);
            println!("Gray analysis: {}", format_analysis(&analysis));
        }

        if let Some(burst_count) = burst_count {
            if burst_count == 0 {
                anyhow::bail!("--burst must be greater than 0");
            }

            println!();
            println!("Burst analysis ({burst_count} frame(s)):");
            let frames = capture_frames(
                &device.path,
                &CaptureConfig {
                    width,
                    height,
                    fourcc,
                    ..CaptureConfig::default()
                },
                burst_count,
            )?;

            for (index, burst_frame) in frames.iter().enumerate() {
                let grayscale = decode_to_grayscale(burst_frame)?;
                let analysis = analyzer.analyze(&grayscale);
                println!(
                    "  frame={} seq={} {}",
                    index,
                    burst_frame.sequence,
                    format_analysis(&analysis)
                );
            }
        }
    }

    Ok(())
}

fn format_device_line(device: &camera::VideoDeviceInfo, primary_symlink: Option<&Path>) -> String {
    let by_path = primary_symlink
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "-".to_string());
    let capture = if device.supports_capture {
        "capture"
    } else {
        "no-capture"
    };
    let streaming = if device.supports_streaming {
        "streaming"
    } else {
        "no-streaming"
    };

    format!(
        "{} card=\"{}\" driver=\"{}\" bus=\"{}\" format={} size={}x{} mode={}/{} by-path={}",
        device.path.display(),
        device.card,
        device.driver,
        device.bus,
        device.fourcc,
        device.width,
        device.height,
        capture,
        streaming,
        by_path
    )
}

fn parse_fourcc(value: &str) -> anyhow::Result<[u8; 4]> {
    let bytes = value.as_bytes();
    if bytes.len() != 4 {
        anyhow::bail!("fourcc must be exactly 4 ASCII bytes");
    }

    Ok([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn default_target_user() -> String {
    if unsafe { libc::geteuid() } == 0 {
        if let Ok(sudo_user) = std::env::var("SUDO_USER") {
            if !sudo_user.trim().is_empty() {
                return sudo_user;
            }
        }
    }
    std::env::var("USER").unwrap_or_else(|_| "unknown".to_string())
}

fn run_verify(args: &[String]) -> anyhow::Result<()> {
    let mut username = default_target_user();
    let mut device: Option<PathBuf> = None;
    let mut frames = 10usize;
    let mut threshold = DEFAULT_THRESHOLD;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--user" => username = iter.next().context("--user requires a value")?.clone(),
            "--device" => device = Some(PathBuf::from(iter.next().context("--device requires a path")?)),
            "--frames" => {
                frames = iter.next().context("--frames requires a value")?
                    .parse().context("--frames must be a positive integer")?;
            }
            "--threshold" => {
                threshold = iter.next().context("--threshold requires a value")?
                    .parse().context("--threshold must be a float")?;
            }
            other => anyhow::bail!("unknown verify argument: {other}"),
        }
    }

    let enrollment = recognition::enrollment::load_enrollment(&username)?
        .with_context(|| format!("no enrollment found for '{username}'; run enroll first"))?;
    let enrolled = enrollment.embeddings_as_arrays();
    if enrolled.is_empty() {
        anyhow::bail!("enrollment for '{username}' has no valid 128-dim embeddings");
    }
    println!("Loaded {} enrolled embedding(s) for '{username}'", enrolled.len());
    println!("Threshold: {threshold:.4}");

    let devices = discover_video_devices()?;
    let ir_device = match device {
        Some(p) => p,
        None => {
            let preferred = preferred_device_from_env();
            let fallback = select_capture_device(&devices, preferred.as_deref());
            devices.iter()
                .find(|d| d.supports_capture && d.supports_streaming
                    && matches!(d.fourcc.as_str(), "GREY" | "Y8"))
                .or_else(|| fallback.as_ref())
                .map(|d| d.path.clone())
                .context("no suitable camera found; use --device")?
        }
    };
    println!("Camera: {}", ir_device.display());

    println!("Loading models...");
    let mut recognizer = load_face_recognizer_blocking()?;
    println!("Ready — hold face in front of camera.\n");
    println!("{:<8} {:<12} {}", "Frame", "Score", "Result");
    println!("{}", "-".repeat(32));

    let cap_config = CaptureConfig::default();
    let mut frame_n = 0usize;
    let mut best: f32 = f32::NEG_INFINITY;

    stream_frames(&ir_device, &cap_config, 2, |frame| {
        if frame_n >= frames {
            return Ok(StreamControl::Stop);
        }
        frame_n += 1;

        let gray = decode_to_grayscale(&frame)?;
        match recognizer.extract(&gray)? {
            None => println!("{:<8} {:<12} no face", frame_n, "-"),
            Some(emb) => {
                let score = FaceRecognizer::match_score(&emb, &enrolled);
                if score > best { best = score; }
                let verdict = if score >= threshold { "MATCH ✓" } else { "no match" };
                println!("{:<8} {:<12.4} {}", frame_n, score, verdict);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        Ok(StreamControl::Continue)
    })?;

    println!("{}", "-".repeat(32));
    if best == f32::NEG_INFINITY {
        println!("Best score: (no face detected)");
    } else {
        let verdict = if best >= threshold { "WOULD ALLOW" } else { "WOULD DENY" };
        println!("Best score: {best:.4}  →  {verdict}  (threshold={threshold:.4})");
    }
    Ok(())
}

fn run_enroll_verify(args: &[String]) -> anyhow::Result<()> {
    let mut username = default_target_user();
    let mut device: Option<PathBuf> = None;
    let mut enroll_frames: usize = 5;
    let mut verify_frames: usize = 10;
    let mut threshold = DEFAULT_THRESHOLD;
    let mut replace = false;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--user" => username = iter.next().context("--user requires a value")?.clone(),
            "--device" => {
                device = Some(PathBuf::from(
                    iter.next().context("--device requires a path")?,
                ))
            }
            "--frames" => {
                enroll_frames = iter
                    .next()
                    .context("--frames requires a value")?
                    .parse()
                    .context("--frames must be a positive integer")?;
            }
            "--verify-frames" => {
                verify_frames = iter
                    .next()
                    .context("--verify-frames requires a value")?
                    .parse()
                    .context("--verify-frames must be a positive integer")?;
            }
            "--threshold" => {
                threshold = iter
                    .next()
                    .context("--threshold requires a value")?
                    .parse()
                    .context("--threshold must be a float")?;
            }
            "--replace" => replace = true,
            other => anyhow::bail!("unknown enroll-verify argument: {other}"),
        }
    }

    let mut enroll_args = vec![
        "--user".to_string(),
        username.clone(),
        "--frames".to_string(),
        enroll_frames.to_string(),
        "--threshold".to_string(),
        threshold.to_string(),
    ];
    if replace {
        enroll_args.push("--replace".to_string());
    }
    if let Some(device) = &device {
        enroll_args.push("--device".to_string());
        enroll_args.push(device.display().to_string());
    }

    println!("=== Step 1/2: enroll ===");
    run_enroll(&enroll_args)?;

    let mut verify_args = vec![
        "--user".to_string(),
        username,
        "--frames".to_string(),
        verify_frames.to_string(),
        "--threshold".to_string(),
        threshold.to_string(),
    ];
    if let Some(device) = &device {
        verify_args.push("--device".to_string());
        verify_args.push(device.display().to_string());
    }

    println!();
    println!("=== Step 2/2: verify ===");
    run_verify(&verify_args)?;
    Ok(())
}

fn run_setup(args: &[String]) -> anyhow::Result<()> {
    let mut username = default_target_user();
    let mut device: Option<PathBuf> = None;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--user" => username = iter.next().context("--user requires a value")?.clone(),
            "--device" => {
                device = Some(PathBuf::from(
                    iter.next().context("--device requires a path")?,
                ))
            }
            other => anyhow::bail!("unknown setup argument: {other}"),
        }
    }

    let mut onboard_args = vec![
        "--user".to_string(),
        username,
        "--replace".to_string(),
    ];
    if let Some(device) = &device {
        onboard_args.push("--device".to_string());
        onboard_args.push(device.display().to_string());
    }

    println!("Running quick setup (enroll + verify) with sensible defaults...");
    run_enroll_verify(&onboard_args)
}

fn print_usage() {
    println!("Usage:");
    println!("  face-authd                         Run the local authentication daemon");
    println!("  face-authd setup                   Quick setup (enroll + verify, recommended)");
    println!("  face-authd enroll [...]            Enroll a user's face for authentication");
    println!("  face-authd enroll-verify [...]     Enroll and verify in one command");
    println!("  face-authd enrolled                List enrolled users");
    println!("  face-authd unenroll [--user USER]  Remove a user's face enrollment");
    println!("  face-authd verify [...]            Test recognition and show confidence scores");
    println!("  face-authd preview [...]           Show a live IR preview window");
    println!("  face-authd probe-cameras [...]     Inspect V4L2 devices");
    println!("  face-authd hardware-survey         Collect camera hardware diagnostics");
    println!();
    println!("quick setup options:");
    println!("  --user USER      Username (default: current user)");
    println!("  --device PATH    IR camera device (default: auto-detect GREY device)");
    println!();
    println!("enroll options:");
    println!("  --user USER      Username to enroll (default: current user)");
    println!("  --device PATH    IR camera device (default: auto-detect GREY device)");
    println!("  --frames N       Number of face samples to collect (default: 5)");
    println!("  --replace        Overwrite existing enrollment for this user");
    println!("  --threshold F    Min match score 0..1 (default: 0.6)");
    println!("  --verify-frames N  (enroll-verify only) verification frames (default: 10)");
    println!();
    print_preview_usage();
    println!();
    print_probe_usage();
}

fn print_preview_usage() {
    println!("preview options:");
    println!("  --device PATH    IR camera device (default: auto-detect GREY device)");
    println!("  --width N        Request capture width");
    println!("  --height N       Request capture height");
    println!("  --fourcc CODE    Request a 4-byte pixel format such as GREY or MJPG");
    println!("  --detect         Run face detection and draw boxes");
    println!("  --frames N       Stop after N streamed frames");
}

fn print_probe_usage() {
    println!("probe-cameras options:");
    println!("  --capture           Attempt a single-frame capture on the selected device");
    println!("  --device PATH       Prefer a specific device or /dev/v4l/by-path symlink");
    println!("  --width N           Request a capture width for --capture");
    println!("  --height N          Request a capture height for --capture");
    println!("  --fourcc CODE       Request a 4-byte pixel format such as YUYV or MJPG");
    println!("  --output PATH       Save the captured frame to PATH");
    println!("  --output-gray PATH  Convert to grayscale and save as PGM");
    println!("  --analyze-gray      Run grayscale quality analysis for the captured frame");
    println!("  --burst N           Capture N additional frames and print per-frame analysis");
    println!("  --list-controls     List V4L2 controls for the selected device");
    println!("  --set-control X=Y   Temporarily set an integer or boolean control");
}

fn run_hardware_survey() -> anyhow::Result<()> {
    println!("Hardware survey");
    println!();

    let devices = discover_video_devices()?;
    println!("V4L2 devices: {}", devices.len());
    let mut visited_usb_roots = BTreeSet::new();
    for device in &devices {
        println!(
            "  {} card=\"{}\" driver=\"{}\" bus=\"{}\" format={} size={}x{} capture={} streaming={}",
            device.path.display(),
            device.card,
            device.driver,
            device.bus,
            device.fourcc,
            device.width,
            device.height,
            device.supports_capture,
            device.supports_streaming
        );
        for link in &device.symlinks {
            println!("    symlink={}", link.display());
        }
        for line in collect_sysfs_details(&device.path)? {
            println!("    {}", line);
        }
        if let Some(usb_root) = usb_device_root_from_video_path(&device.path)? {
            if visited_usb_roots.insert(usb_root.clone()) {
                for line in collect_usb_interface_details(&usb_root)? {
                    println!("    {}", line);
                }
            }
        }
    }

    println!();
    println!("hidraw nodes:");
    match list_glob_prefix("/dev", "hidraw")? {
        nodes if nodes.is_empty() => println!("  none"),
        nodes => {
            for node in nodes {
                println!("  {}", node.display());
            }
        }
    }

    println!();
    println!("lsusb:");
    print_command_report("lsusb", &mut Command::new("lsusb"))?;

    println!();
    println!("dmesg (camera-related tail):");
    let mut dmesg = Command::new("dmesg");
    dmesg.arg("--color=never");
    print_command_report("dmesg", &mut dmesg)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Preview frame sent from camera thread to GUI thread
// ---------------------------------------------------------------------------

struct PreviewFrame {
    rgb: Vec<u8>,
    width: usize,
    height: usize,
    detections: Vec<Detection>,
}

// ---------------------------------------------------------------------------
// egui application
// ---------------------------------------------------------------------------

struct PreviewApp {
    rx: std::sync::mpsc::Receiver<PreviewFrame>,
    texture: Option<egui::TextureHandle>,
    frame_w: usize,
    frame_h: usize,
    detections: Vec<Detection>,
    sized_to_frame: bool,
}

impl eframe::App for PreviewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain all pending frames, keep the latest
        while let Ok(f) = self.rx.try_recv() {
            let image = egui::ColorImage::from_rgb([f.width, f.height], &f.rgb);
            self.texture = Some(ctx.load_texture(
                "camera",
                image,
                egui::TextureOptions::NEAREST,
            ));
            // Resize window to match camera resolution on first frame
            if !self.sized_to_frame || self.frame_w != f.width || self.frame_h != f.height {
                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(
                    egui::vec2(f.width as f32, f.height as f32),
                ));
                self.sized_to_frame = true;
            }
            self.frame_w = f.width;
            self.frame_h = f.height;
            self.detections = f.detections;
        }

        // Remove egui panel padding so image fills the window exactly
        let frame = egui::Frame::none();
        egui::CentralPanel::default().frame(frame).show(ctx, |ui| {
            if let Some(tex) = &self.texture {
                let available = ui.available_size();
                let resp = ui.add(egui::Image::new(tex).fit_to_exact_size(available));

                // Draw detection overlays with egui painter
                if self.frame_w > 0 && self.frame_h > 0 {
                    let r = resp.rect;
                    let sx = r.width()  / self.frame_w as f32;
                    let sy = r.height() / self.frame_h as f32;
                    let painter = ui.painter();

                    for det in &self.detections {
                        let x0 = r.left() + det.x * sx;
                        let y0 = r.top()  + det.y * sy;
                        let x1 = x0 + det.w * sx;
                        let y1 = y0 + det.h * sy;
                        let drect = egui::Rect::from_min_max(
                            egui::pos2(x0, y0), egui::pos2(x1, y1),
                        );
                        painter.rect_stroke(
                            drect,
                            egui::Rounding::ZERO,
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 80, 0)),
                        );
                        painter.text(
                            egui::pos2(x0 + 2.0, y0 + 2.0),
                            egui::Align2::LEFT_TOP,
                            format!("{:.2}", det.score),
                            egui::FontId::monospace(13.0),
                            egui::Color32::from_rgb(255, 220, 0),
                        );
                    }
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Waiting for camera…");
                });
            }
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(33));
    }
}

// ---------------------------------------------------------------------------
// run_preview
// ---------------------------------------------------------------------------

fn run_preview(args: &[String]) -> anyhow::Result<()> {
    let mut device: Option<PathBuf> = None;
    let mut width = None;
    let mut height = None;
    let mut fourcc = None;
    let mut detect = false;
    let mut max_frames: Option<usize> = None;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--device" => {
                device = Some(PathBuf::from(
                    iter.next().context("--device requires a path")?,
                ));
            }
            "--width" => {
                width = Some(
                    iter.next()
                        .context("--width requires a value")?
                        .parse()
                        .context("--width must be an integer")?,
                );
            }
            "--height" => {
                height = Some(
                    iter.next()
                        .context("--height requires a value")?
                        .parse()
                        .context("--height must be an integer")?,
                );
            }
            "--fourcc" => {
                fourcc = Some(parse_fourcc(
                    iter.next().context("--fourcc requires a 4-byte code")?,
                )?);
            }
            "--detect" => detect = true,
            "--frames" => {
                max_frames = Some(
                    iter.next()
                        .context("--frames requires a value")?
                        .parse()
                        .context("--frames must be a positive integer")?,
                );
            }
            "-h" | "--help" => {
                print_preview_usage();
                return Ok(());
            }
            other => anyhow::bail!("unknown preview argument: {other}"),
        }
    }

    let devices = discover_video_devices()?;
    let selected = match device {
        Some(path) => devices
            .iter()
            .find(|d| d.path == path || d.symlinks.iter().any(|link| link == &path))
            .cloned()
            .map(|mut d| { d.path = path.clone(); d })
            .unwrap_or(camera::VideoDeviceInfo {
                path,
                symlinks: Vec::new(),
                driver: String::new(),
                card: String::new(),
                bus: String::new(),
                supports_capture: true,
                supports_streaming: true,
                width: width.unwrap_or(640),
                height: height.unwrap_or(360),
                fourcc: fourcc
                    .map(|fcc| String::from_utf8_lossy(&fcc).to_string())
                    .unwrap_or_else(|| "GREY".to_string()),
            }),
        None => {
            let preferred = preferred_device_from_env();
            let fallback = select_capture_device(&devices, preferred.as_deref());
            devices
                .iter()
                .find(|d| {
                    d.supports_capture
                        && d.supports_streaming
                        && matches!(d.fourcc.as_str(), "GREY" | "Y8")
                })
                .or_else(|| fallback.as_ref())
                .cloned()
                .context("no suitable preview camera found; use --device")?
        }
    };
    let ir_device = selected.path.clone();
    println!("Preview device: {}", ir_device.display());

    let config = CaptureConfig { width, height, fourcc, ..CaptureConfig::default() };

    // Channel: camera thread → GUI thread
    let (tx, rx) = std::sync::mpsc::channel::<PreviewFrame>();

    // Spawn camera thread
    std::thread::spawn(move || {
        let mut detector: Option<FaceDetector> = if detect {
            match load_face_detector_blocking() {
                Ok(d) => Some(d),
                Err(e) => { eprintln!("detector load failed: {e}"); None }
            }
        } else {
            None
        };

        let mut streamed = 0usize;
        let _ = stream_frames(&ir_device, &config, 2, |frame| {
            let gray = decode_to_grayscale(&frame)?;

            let detections = match detector.as_mut() {
                Some(d) => d.detect(&gray).unwrap_or_default(),
                None    => vec![],
            };

            // Grayscale → RGB24
            let rgb: Vec<u8> = gray.pixels.iter().flat_map(|&p| [p, p, p]).collect();

            if tx.send(PreviewFrame {
                rgb,
                width:  gray.width  as usize,
                height: gray.height as usize,
                detections,
            }).is_err() {
                // Receiver dropped (window closed)
                return Ok(StreamControl::Stop);
            }

            streamed += 1;
            if max_frames.is_some_and(|limit| streamed >= limit) {
                return Ok(StreamControl::Stop);
            }
            Ok(StreamControl::Continue)
        });
    });

    // Run GUI on the main thread
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("face-authd preview")
            .with_inner_size([640.0, 480.0]),
        ..Default::default()
    };
    eframe::run_native(
        "face-authd preview",
        options,
        Box::new(|cc| {
            // Light visuals to better match Ubuntu GNOME
            cc.egui_ctx.set_visuals(egui::Visuals::light());
            Ok(Box::new(PreviewApp {
                rx,
                texture: None,
                frame_w: 0,
                frame_h: 0,
                detections: vec![],
                sized_to_frame: false,
            }))
        }),
    ).map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;

    Ok(())
}

fn collect_sysfs_details(device_path: &Path) -> anyhow::Result<Vec<String>> {
    let name = device_path
        .file_name()
        .with_context(|| format!("missing filename for {}", device_path.display()))?
        .to_string_lossy()
        .to_string();
    let sys_dir = PathBuf::from("/sys/class/video4linux").join(name);
    let real_path = fs::canonicalize(&sys_dir)
        .with_context(|| format!("failed to resolve sysfs path {}", sys_dir.display()))?;

    let mut lines = Vec::new();
    lines.push(format!("sysfs={}", real_path.display()));

    for (label, path) in [
        ("index", sys_dir.join("index")),
        ("modalias", sys_dir.join("device/modalias")),
        ("interface", sys_dir.join("device/interface")),
        ("driver", sys_dir.join("device/driver/module")),
    ] {
        if let Ok(value) = fs::read_to_string(&path) {
            lines.push(format!("{}={}", label, value.trim()));
        } else if label == "driver" {
            if let Ok(link) = fs::read_link(&path) {
                lines.push(format!("{}={}", label, link.display()));
            }
        }
    }

    Ok(lines)
}

fn usb_device_root_from_video_path(device_path: &Path) -> anyhow::Result<Option<PathBuf>> {
    let name = device_path
        .file_name()
        .with_context(|| format!("missing filename for {}", device_path.display()))?
        .to_string_lossy()
        .to_string();
    let sys_dir = PathBuf::from("/sys/class/video4linux").join(name);
    let real_path = fs::canonicalize(&sys_dir)
        .with_context(|| format!("failed to resolve sysfs path {}", sys_dir.display()))?;

    for ancestor in real_path.ancestors() {
        if let Some(name) = ancestor.file_name().and_then(|name| name.to_str()) {
            if name.starts_with("3-") && !name.contains(':') {
                return Ok(Some(ancestor.to_path_buf()));
            }
        }
    }

    Ok(None)
}

fn collect_usb_interface_details(usb_root: &Path) -> anyhow::Result<Vec<String>> {
    let mut lines = Vec::new();
    lines.push(format!("usb-root={}", usb_root.display()));

    let mut interfaces = Vec::new();
    for entry in fs::read_dir(usb_root)
        .with_context(|| format!("failed to read {}", usb_root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with("3-") && name.contains(':') {
            interfaces.push(path);
        }
    }
    interfaces.sort();

    for interface in interfaces {
        let name = interface
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        let class = fs::read_to_string(interface.join("bInterfaceClass"))
            .ok()
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|| "-".to_string());
        let subclass = fs::read_to_string(interface.join("bInterfaceSubClass"))
            .ok()
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|| "-".to_string());
        let protocol = fs::read_to_string(interface.join("bInterfaceProtocol"))
            .ok()
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|| "-".to_string());
        let label = fs::read_to_string(interface.join("interface"))
            .ok()
            .map(|value| value.trim().to_string())
            .unwrap_or_else(|| classify_usb_interface(&class, &subclass));
        lines.push(format!(
            "usb-interface={} class={} subclass={} protocol={} label=\"{}\"",
            name, class, subclass, protocol, label
        ));
    }

    Ok(lines)
}

fn classify_usb_interface(class: &str, subclass: &str) -> String {
    match (class, subclass) {
        ("0e", "01") => "VideoStreaming".to_string(),
        ("0e", "02") => "VideoControl".to_string(),
        ("fe", "01") => "ApplicationSpecific".to_string(),
        _ => "Unknown".to_string(),
    }
}

fn list_glob_prefix(base: &str, prefix: &str) -> anyhow::Result<Vec<PathBuf>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(base).with_context(|| format!("failed to read {}", base))? {
        let entry = entry?;
        let path = entry.path();
        if path
            .file_name()
            .map(|name| name.to_string_lossy().starts_with(prefix))
            .unwrap_or(false)
        {
            entries.push(path);
        }
    }
    entries.sort();
    Ok(entries)
}

fn run_enroll(args: &[String]) -> anyhow::Result<()> {
    let mut username = default_target_user();
    let mut device: Option<PathBuf> = None;
    let mut target_frames: usize = 5;
    let mut replace_existing = false;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--user" => {
                username = iter.next().context("--user requires a value")?.clone();
            }
            "--device" => {
                device = Some(PathBuf::from(
                    iter.next().context("--device requires a path")?,
                ));
            }
            "--frames" => {
                target_frames = iter
                    .next()
                    .context("--frames requires a value")?
                    .parse()
                    .context("--frames must be a positive integer")?;
            }
            "--replace" => {
                replace_existing = true;
            }
            "--threshold" => {
                let _det_threshold: f32 = iter
                    .next()
                    .context("--threshold requires a value")?
                    .parse()
                    .context("--threshold must be a float")?;
            }
            other => anyhow::bail!("unknown enroll argument: {other}"),
        }
    }

    println!("Enrolling user: {username}");

    // Ensure enrollment dir is accessible
    let enroll_dir = enrollment_dir();
    check_enrollment_dir(&enroll_dir)
        .with_context(|| format!("enrollment directory {} is not usable", enroll_dir.display()))?;

    if load_enrollment(&username)?.is_some() && !replace_existing {
        anyhow::bail!(
            "enrollment for '{}' already exists; use --replace to overwrite or run `face-authd unenroll --user {}` first",
            username,
            username
        );
    }

    // Select IR camera
    let devices = discover_video_devices()?;
    let ir_device = match device {
        Some(ref p) => p.clone(),
        None => {
            let preferred = preferred_device_from_env();
            let fallback = select_capture_device(&devices, preferred.as_deref());
            let found = devices
                .iter()
                .find(|d| d.supports_capture && d.supports_streaming
                    && matches!(d.fourcc.as_str(), "GREY" | "Y8"))
                .or_else(|| fallback.as_ref())
                .map(|d| d.path.clone())
                .context("no suitable camera found; use --device")?;
            found
        }
    };
    println!("IR camera: {}", ir_device.display());

    // Load models (may trigger download)
    println!("Loading recognition models...");
    let mut recognizer =
        load_face_recognizer_blocking().context("failed to load face recognition models")?;
    println!("Models ready.");

    let cap_config = CaptureConfig::default();
    let mut collected: Vec<Vec<f32>> = Vec::new();
    let mut attempt = 0usize;
    let max_attempts = target_frames * 10;

    println!(
        "Position your face in front of the IR camera. Collecting {target_frames} samples..."
    );

    stream_frames(&ir_device, &cap_config, 2, |frame| {
        if collected.len() >= target_frames || attempt >= max_attempts {
            return Ok(StreamControl::Stop);
        }
        attempt += 1;

        let gray = match decode_to_grayscale(&frame) {
            Ok(g) => g,
            Err(err) => {
                eprintln!("  decode error: {err}");
                std::thread::sleep(std::time::Duration::from_millis(200));
                return Ok(StreamControl::Continue);
            }
        };

        match recognizer.extract(&gray) {
            Ok(Some(embedding)) => {
                collected.push(embedding.to_vec());
                println!("  [{}/{}] face captured", collected.len(), target_frames);
                if collected.len() >= target_frames {
                    return Ok(StreamControl::Stop);
                }
            }
            Ok(None) => {
                eprint!(".");
            }
            Err(err) => {
                eprintln!("  recognition error: {err:#}");
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(150));
        Ok(StreamControl::Continue)
    })
    .or_else(|err| {
        eprintln!("  capture error: {err}");
        Err(err)
    })?;

    if collected.len() < target_frames {
        anyhow::bail!(
            "only {}/{} samples collected after {attempt} attempts; ensure IR emitter is active and face is visible",
            collected.len(),
            target_frames
        );
    }

    let enrollment = UserEnrollment {
        username: username.clone(),
        embeddings: collected,
    };
    save_enrollment(&enrollment)?;
    println!(
        "Enrolled {username} with {} samples → {}",
        enrollment.embeddings.len(),
        recognition::enrollment::enrollment_path(&username).display()
    );

    Ok(())
}

fn run_enrolled() -> anyhow::Result<()> {
    let users = list_enrolled_users()?;
    if users.is_empty() {
        println!("No enrolled users.");
    } else {
        println!("Enrolled users ({}):", users.len());
        for user in &users {
            if let Ok(Some(e)) = load_enrollment(user) {
                println!("  {} ({} sample(s))", user, e.embeddings.len());
            }
        }
    }
    Ok(())
}

fn run_unenroll(args: &[String]) -> anyhow::Result<()> {
    let mut username = default_target_user();

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--user" => {
                username = iter.next().context("--user requires a value")?.clone();
            }
            other => anyhow::bail!("unknown unenroll argument: {other}"),
        }
    }

    if delete_enrollment(&username)? {
        println!("Removed enrollment for {username}.");
    } else {
        println!("No enrollment found for {username}.");
    }
    Ok(())
}

fn print_command_report(name: &str, command: &mut Command) -> anyhow::Result<()> {
    match command.output() {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.trim().is_empty() {
                    println!("  no output");
                } else {
                    for line in stdout.lines().take(120) {
                        println!("  {}", line);
                    }
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                println!(
                    "  command failed (exit={}): {}",
                    output.status,
                    stderr.trim()
                );
            }
        }
        Err(err) => {
            println!("  unavailable: {}", err);
        }
    }
    let _ = name;
    Ok(())
}

async fn handle_client(
    stream: UnixStream,
    provider: Arc<dyn FaceAuthProvider>,
) -> anyhow::Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();
    reader.read_line(&mut line).await?;

    let response = match serde_json::from_str::<Request>(&line) {
        Ok(Request::Ping) => Response::Pong,
        Ok(Request::Authenticate(req)) => authenticate(provider.as_ref(), req).await,
        Err(err) => Response::Error(ErrorResponse {
            code: "bad_request".to_string(),
            message: err.to_string(),
        }),
    };

    let payload = encode_response(&response)?;
    writer.write_all(&payload).await?;
    writer.flush().await?;
    Ok(())
}

async fn authenticate(provider: &dyn FaceAuthProvider, req: AuthenticateRequest) -> Response {
    if req.version != PROTOCOL_VERSION {
        return Response::Error(ErrorResponse {
            code: "protocol_version_mismatch".to_string(),
            message: format!("expected version {}, got {}", PROTOCOL_VERSION, req.version),
        });
    }

    info!(
        username = req.username,
        service = req.service.as_deref().unwrap_or("unknown"),
        tty = req.tty.as_deref().unwrap_or("unknown"),
        "received authentication request"
    );

    match provider.authenticate(&req).await {
        AuthDecision::Allow { reason } => Response::Authenticate(AuthenticateResponse {
            success: true,
            reason,
        }),
        AuthDecision::Deny { reason } => {
            warn!(
                username = req.username,
                reason = reason.as_deref().unwrap_or("none"),
                "authentication denied"
            );
            Response::Authenticate(AuthenticateResponse {
                success: false,
                reason,
            })
        }
        AuthDecision::Error { message } => Response::Error(ErrorResponse {
            code: "provider_error".to_string(),
            message,
        }),
    }
}
