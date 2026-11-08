use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use jpeg_decoder::{Decoder as JpegDecoder, PixelFormat as JpegPixelFormat};
use v4l::buffer::Type;
use v4l::capability::Flags;
use v4l::format::FourCC;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::Device;

#[derive(Debug, Clone)]
pub struct VideoDeviceInfo {
    pub path: PathBuf,
    pub symlinks: Vec<PathBuf>,
    pub driver: String,
    pub card: String,
    pub bus: String,
    pub supports_capture: bool,
    pub supports_streaming: bool,
    pub width: u32,
    pub height: u32,
    pub fourcc: String,
}

#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fourcc: Option<[u8; 4]>,
    pub buffers: u32,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            fourcc: None,
            buffers: 4,
        }
    }
}

/// Keep enrollment, verification, and daemon capture resolution consistent.
pub fn auth_capture_config() -> CaptureConfig {
    CaptureConfig {
        width: Some(640),
        height: Some(360),
        ..Default::default()
    }
}

#[derive(Debug, Clone)]
pub struct CapturedFrame {
    pub device: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fourcc: String,
    pub sequence: u32,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct GrayscaleFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamControl {
    Continue,
    Stop,
}

impl GrayscaleFrame {
    pub fn validate(&self) -> Result<()> {
        let length = (self.width as usize).checked_mul(self.height as usize);
        anyhow::ensure!(
            self.width > 0
                && self.height > 0
                && self.width <= i32::MAX as u32
                && self.height <= i32::MAX as u32
                && length == Some(self.pixels.len()),
            "invalid grayscale frame dimensions or buffer length"
        );
        Ok(())
    }
}

pub fn discover_video_devices() -> Result<Vec<VideoDeviceInfo>> {
    let symlink_map = collect_v4l_symlinks("/dev/v4l/by-path");
    let mut devices = Vec::new();

    for entry in fs::read_dir("/sys/class/video4linux")
        .context("failed to enumerate /sys/class/video4linux")?
    {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };

        let dev_path = PathBuf::from("/dev").join(name);
        if !dev_path.exists() {
            continue;
        }

        let device = match Device::with_path(&dev_path) {
            Ok(device) => device,
            Err(_) => continue,
        };

        let caps = match device.query_caps() {
            Ok(caps) => caps,
            Err(_) => continue,
        };

        let format = match device.format() {
            Ok(format) => format,
            Err(_) => continue,
        };

        let fourcc = fourcc_to_string(format.fourcc);
        let symlinks = symlink_map.get(&dev_path).cloned().unwrap_or_default();
        let supports_capture = caps.capabilities.contains(Flags::VIDEO_CAPTURE)
            || caps.capabilities.contains(Flags::VIDEO_CAPTURE_MPLANE);
        let supports_streaming = caps.capabilities.contains(Flags::STREAMING);

        devices.push(VideoDeviceInfo {
            path: dev_path,
            symlinks,
            driver: caps.driver,
            card: caps.card,
            bus: caps.bus,
            supports_capture,
            supports_streaming,
            width: format.width,
            height: format.height,
            fourcc,
        });
    }

    devices.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(devices)
}

pub fn select_capture_device(
    devices: &[VideoDeviceInfo],
    preferred: Option<&Path>,
) -> Option<VideoDeviceInfo> {
    if let Some(preferred) = preferred {
        let preferred = preferred.to_path_buf();
        if let Some(found) = devices.iter().find(|device| {
            device.path == preferred || device.symlinks.iter().any(|link| link == &preferred)
        }) {
            return Some(found.clone());
        }
    }

    devices
        .iter()
        .find(|device| device.supports_capture && device.supports_streaming)
        .cloned()
        .or_else(|| devices.first().cloned())
}

/// An explicit stable path wins. Never silently switch cameras if it disappears.
pub fn select_auth_device(devices: &[VideoDeviceInfo]) -> Result<VideoDeviceInfo> {
    select_auth_device_with_preference(devices, preferred_device_from_env().as_deref())
}

fn select_auth_device_with_preference(
    devices: &[VideoDeviceInfo],
    preferred: Option<&Path>,
) -> Result<VideoDeviceInfo> {
    if let Some(preferred) = preferred {
        let preferred = preferred.to_path_buf();
        let canonical = fs::canonicalize(&preferred).unwrap_or(preferred.clone());
        return devices
            .iter()
            .find(|d| {
                d.supports_capture
                    && d.supports_streaming
                    && (d.path == canonical
                        || d.path == preferred
                        || d.symlinks.contains(&preferred))
            })
            .cloned()
            .with_context(|| format!("configured camera {} is unavailable", preferred.display()));
    }
    devices
        .iter()
        .find(|d| {
            d.supports_capture && d.supports_streaming && matches!(d.fourcc.as_str(), "GREY" | "Y8")
        })
        .cloned()
        .or_else(|| {
            devices
                .iter()
                .find(|d| d.supports_capture && d.supports_streaming)
                .cloned()
        })
        .context("no capture camera found")
}

pub fn capture_single_frame(device_path: &Path, config: &CaptureConfig) -> Result<CapturedFrame> {
    capture_frames(device_path, config, 1)?
        .into_iter()
        .next()
        .context("capture returned no frames")
}

pub fn capture_frames(
    device_path: &Path,
    config: &CaptureConfig,
    count: usize,
) -> Result<Vec<CapturedFrame>> {
    let dev = Device::with_path(device_path)
        .with_context(|| format!("failed to open video device {}", device_path.display()))?;

    let mut format = dev
        .format()
        .with_context(|| format!("failed to read device format for {}", device_path.display()))?;

    if let Some(width) = config.width {
        format.width = width;
    }
    if let Some(height) = config.height {
        format.height = height;
    }
    if let Some(fourcc) = config.fourcc {
        format.fourcc = FourCC::new(&fourcc);
    }

    let format = dev
        .set_format(&format)
        .with_context(|| format!("failed to set format on {}", device_path.display()))?;

    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, config.buffers)
        .with_context(|| format!("failed to create mmap stream for {}", device_path.display()))?;

    stream.set_timeout(std::time::Duration::from_secs(2));
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        let (buf, meta) = stream
            .next()
            .with_context(|| format!("failed to capture frame from {}", device_path.display()))?;

        frames.push(CapturedFrame {
            device: device_path.to_path_buf(),
            width: format.width,
            height: format.height,
            fourcc: fourcc_to_string(format.fourcc),
            sequence: meta.sequence,
            bytes: buf.to_vec(),
        });
    }

    Ok(frames)
}

pub fn stream_frames<F>(
    device_path: &Path,
    config: &CaptureConfig,
    warmup_frames: usize,
    mut on_frame: F,
) -> Result<usize>
where
    F: FnMut(CapturedFrame) -> Result<StreamControl>,
{
    let dev = Device::with_path(device_path)
        .with_context(|| format!("failed to open video device {}", device_path.display()))?;

    let mut format = dev
        .format()
        .with_context(|| format!("failed to read device format for {}", device_path.display()))?;

    if let Some(width) = config.width {
        format.width = width;
    }
    if let Some(height) = config.height {
        format.height = height;
    }
    if let Some(fourcc) = config.fourcc {
        format.fourcc = FourCC::new(&fourcc);
    }

    let format = dev
        .set_format(&format)
        .with_context(|| format!("failed to set format on {}", device_path.display()))?;

    let mut stream = MmapStream::with_buffers(&dev, Type::VideoCapture, config.buffers)
        .with_context(|| format!("failed to create mmap stream for {}", device_path.display()))?;

    stream.set_timeout(std::time::Duration::from_millis(500));
    for _ in 0..warmup_frames {
        let _ = stream
            .next()
            .with_context(|| format!("failed to warm up stream on {}", device_path.display()))?;
    }

    let mut processed = 0usize;
    loop {
        let (buf, meta) = stream
            .next()
            .with_context(|| format!("failed to capture frame from {}", device_path.display()))?;

        let frame = CapturedFrame {
            device: device_path.to_path_buf(),
            width: format.width,
            height: format.height,
            fourcc: fourcc_to_string(format.fourcc),
            sequence: meta.sequence,
            bytes: buf.to_vec(),
        };

        processed += 1;
        match on_frame(frame)? {
            StreamControl::Continue => continue,
            StreamControl::Stop => break,
        }
    }

    Ok(processed)
}

pub fn preferred_device_from_env() -> Option<PathBuf> {
    std::env::var_os("FACE_AUTHD_VIDEO_DEVICE").map(PathBuf::from)
}

pub fn write_frame_to_path(frame: &CapturedFrame, output_path: &Path) -> Result<&'static str> {
    match frame.fourcc.as_str() {
        "MJPG" => {
            fs::write(output_path, &frame.bytes).with_context(|| {
                format!("failed to write jpeg frame to {}", output_path.display())
            })?;
            Ok("jpeg")
        }
        "GREY" | "Y8" => {
            let expected = frame.width as usize * frame.height as usize;
            if frame.bytes.len() < expected {
                anyhow::bail!(
                    "grey frame buffer too small: got {} bytes, expected at least {}",
                    frame.bytes.len(),
                    expected
                );
            }

            let mut data = format!("P5\n{} {}\n255\n", frame.width, frame.height).into_bytes();
            data.extend_from_slice(&frame.bytes[..expected]);
            fs::write(output_path, data).with_context(|| {
                format!("failed to write pgm frame to {}", output_path.display())
            })?;
            Ok("pgm")
        }
        _ => {
            fs::write(output_path, &frame.bytes).with_context(|| {
                format!("failed to write raw frame to {}", output_path.display())
            })?;
            Ok("raw")
        }
    }
}

pub fn decode_to_grayscale(frame: &CapturedFrame) -> Result<GrayscaleFrame> {
    match normalized_fourcc(&frame.fourcc).as_str() {
        "GREY" | "Y8" => decode_grey(frame),
        "MJPG" => decode_mjpg(frame),
        "YUYV" | "YUY2" => decode_yuyv(frame),
        other => anyhow::bail!("unsupported pixel format for grayscale conversion: {other}"),
    }
}

pub fn write_grayscale_pgm(frame: &GrayscaleFrame, output_path: &Path) -> Result<()> {
    let expected = frame.width as usize * frame.height as usize;
    if frame.pixels.len() != expected {
        anyhow::bail!(
            "grayscale frame size mismatch: got {} bytes, expected {}",
            frame.pixels.len(),
            expected
        );
    }

    let mut data = format!("P5\n{} {}\n255\n", frame.width, frame.height).into_bytes();
    data.extend_from_slice(&frame.pixels);
    fs::write(output_path, data)
        .with_context(|| format!("failed to write grayscale pgm to {}", output_path.display()))?;
    Ok(())
}

fn collect_v4l_symlinks(base: &str) -> HashMap<PathBuf, Vec<PathBuf>> {
    let mut map = HashMap::new();
    let Ok(entries) = fs::read_dir(base) else {
        return map;
    };

    for entry in entries.flatten() {
        let link_path = entry.path();
        let Ok(target) = fs::canonicalize(&link_path) else {
            continue;
        };
        map.entry(target).or_insert_with(Vec::new).push(link_path);
    }

    map
}

fn fourcc_to_string(fourcc: FourCC) -> String {
    match fourcc.str() {
        Ok(value) => value.to_string(),
        Err(_) => "UNKNOWN".to_string(),
    }
}

fn normalized_fourcc(fourcc: &str) -> String {
    fourcc.trim().to_string()
}

fn decode_grey(frame: &CapturedFrame) -> Result<GrayscaleFrame> {
    let expected = frame.width as usize * frame.height as usize;
    if frame.bytes.len() < expected {
        anyhow::bail!(
            "grey frame buffer too small: got {} bytes, expected at least {}",
            frame.bytes.len(),
            expected
        );
    }

    Ok(GrayscaleFrame {
        width: frame.width,
        height: frame.height,
        pixels: frame.bytes[..expected].to_vec(),
    })
}

fn decode_mjpg(frame: &CapturedFrame) -> Result<GrayscaleFrame> {
    let mut decoder = JpegDecoder::new(std::io::Cursor::new(&frame.bytes));
    let pixels = decoder
        .decode()
        .context("failed to decode mjpg frame as jpeg")?;
    let info = decoder
        .info()
        .context("jpeg decoder did not return image metadata")?;

    let grayscale = match info.pixel_format {
        JpegPixelFormat::L8 => pixels,
        JpegPixelFormat::RGB24 => pixels
            .as_chunks::<3>()
            .0
            .iter()
            .map(|chunk| rgb_to_luma(chunk[0], chunk[1], chunk[2]))
            .collect(),
        JpegPixelFormat::CMYK32 => pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| {
                let c = chunk[0] as f32 / 255.0;
                let m = chunk[1] as f32 / 255.0;
                let y = chunk[2] as f32 / 255.0;
                let k = chunk[3] as f32 / 255.0;
                let r = ((1.0 - c) * (1.0 - k) * 255.0).round() as u8;
                let g = ((1.0 - m) * (1.0 - k) * 255.0).round() as u8;
                let b = ((1.0 - y) * (1.0 - k) * 255.0).round() as u8;
                rgb_to_luma(r, g, b)
            })
            .collect(),
        _ => anyhow::bail!("unsupported jpeg pixel format"),
    };

    Ok(GrayscaleFrame {
        width: info.width.into(),
        height: info.height.into(),
        pixels: grayscale,
    })
}

fn decode_yuyv(frame: &CapturedFrame) -> Result<GrayscaleFrame> {
    let expected = frame.width as usize * frame.height as usize * 2;
    if frame.bytes.len() < expected {
        anyhow::bail!(
            "yuyv frame buffer too small: got {} bytes, expected at least {}",
            frame.bytes.len(),
            expected
        );
    }

    let mut pixels = Vec::with_capacity(frame.width as usize * frame.height as usize);
    for chunk in frame.bytes[..expected].as_chunks::<4>().0.iter() {
        pixels.push(chunk[0]);
        pixels.push(chunk[2]);
    }

    Ok(GrayscaleFrame {
        width: frame.width,
        height: frame.height,
        pixels,
    })
}

fn rgb_to_luma(r: u8, g: u8, b: u8) -> u8 {
    let r = r as u32;
    let g = g as u32;
    let b = b as u32;
    ((299 * r + 587 * g + 114 * b) / 1000) as u8
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    fn device(path: &str, fourcc: &str) -> VideoDeviceInfo {
        VideoDeviceInfo {
            path: path.into(),
            symlinks: vec![],
            driver: String::new(),
            card: String::new(),
            bus: String::new(),
            supports_capture: true,
            supports_streaming: true,
            width: 640,
            height: 360,
            fourcc: fourcc.into(),
        }
    }
    #[test]
    fn preference_wins_over_gray_and_missing_device_fails() {
        let devices = vec![
            device("/dev/mock-rgb", "MJPG"),
            device("/dev/mock-ir", "GREY"),
        ];
        assert_eq!(
            select_auth_device_with_preference(&devices, None)
                .unwrap()
                .path,
            Path::new("/dev/mock-ir")
        );
        assert_eq!(
            select_auth_device_with_preference(&devices, Some(Path::new("/dev/mock-rgb")))
                .unwrap()
                .path,
            Path::new("/dev/mock-rgb")
        );
        assert!(
            select_auth_device_with_preference(&devices, Some(Path::new("/dev/missing"))).is_err()
        );
    }
    #[test]
    fn metadata_nodes_and_invalid_frame_buffers_are_rejected() {
        let mut metadata = device("/dev/metadata", "GREY");
        metadata.supports_capture = false;
        assert!(select_auth_device_with_preference(&[metadata], None).is_err());
        for frame in [
            GrayscaleFrame {
                width: 0,
                height: 1,
                pixels: vec![],
            },
            GrayscaleFrame {
                width: 100,
                height: 100,
                pixels: vec![0; 10],
            },
        ] {
            assert!(frame.validate().is_err());
        }
    }
}

/// Recognition keeps RGB camera color; IR gray pixels are repeated into B,G,R.
#[derive(Debug, Clone)]
pub struct BgrFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}
impl BgrFrame {
    pub fn validate(&self) -> Result<()> {
        let length = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(3));
        anyhow::ensure!(
            self.width > 0
                && self.height > 0
                && self.width <= 8192
                && self.height <= 8192
                && length == Some(self.pixels.len()),
            "invalid BGR frame dimensions or buffer length"
        );
        Ok(())
    }
}
pub fn decode_to_bgr(frame: &CapturedFrame) -> Result<BgrFrame> {
    anyhow::ensure!(
        frame.width > 0 && frame.height > 0 && frame.width <= 8192 && frame.height <= 8192,
        "invalid capture dimensions"
    );
    let count = frame.width as usize * frame.height as usize;
    let pixels = match normalized_fourcc(&frame.fourcc).as_str() {
        "GREY" | "Y8" => {
            let gray = decode_grey(frame)?;
            gray.pixels.iter().flat_map(|v| [*v; 3]).collect()
        }
        "MJPG" | "JPEG" => {
            let mut decoder = JpegDecoder::new(std::io::Cursor::new(&frame.bytes));
            let pixels = decoder.decode().context("invalid JPEG capture")?;
            let info = decoder.info().context("missing JPEG metadata")?;
            anyhow::ensure!(
                u32::from(info.width) == frame.width && u32::from(info.height) == frame.height,
                "JPEG dimensions differ from negotiated capture format"
            );
            match info.pixel_format {
                JpegPixelFormat::RGB24 => pixels
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .flat_map(|p| [p[2], p[1], p[0]])
                    .collect(),
                JpegPixelFormat::L8 => pixels.iter().flat_map(|v| [*v; 3]).collect(),
                _ => anyhow::bail!("unsupported recognition JPEG pixel format"),
            }
        }
        "YUYV" | "YUY2" => {
            anyhow::ensure!(
                frame.width.is_multiple_of(2) && frame.bytes.len() >= count * 2,
                "invalid YUYV frame"
            );
            frame.bytes[..count * 2]
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    let yuv = |y: u8| {
                        let c = i32::from(y) - 16;
                        let d = i32::from(p[1]) - 128;
                        let e = i32::from(p[3]) - 128;
                        let byte = |v: i32| ((v + 128) >> 8).clamp(0, 255) as u8;
                        [
                            byte(298 * c + 516 * d),
                            byte(298 * c - 100 * d - 208 * e),
                            byte(298 * c + 409 * e),
                        ]
                    };
                    let a = yuv(p[0]);
                    let b = yuv(p[2]);
                    [a[0], a[1], a[2], b[0], b[1], b[2]]
                })
                .collect()
        }
        other => anyhow::bail!("unsupported recognition capture format: {other}"),
    };
    let bgr = BgrFrame {
        width: frame.width,
        height: frame.height,
        pixels,
    };
    bgr.validate()?;
    Ok(bgr)
}

#[cfg(test)]
mod bgr_tests {
    use super::*;
    fn capture(fourcc: &str, bytes: Vec<u8>) -> CapturedFrame {
        CapturedFrame {
            device: "/dev/test".into(),
            width: 2,
            height: 1,
            fourcc: fourcc.into(),
            sequence: 1,
            bytes,
        }
    }
    #[test]
    fn gray_and_yuyv_preprocessing() {
        let gray = decode_to_bgr(&capture("GREY", vec![20, 40])).unwrap();
        assert_eq!(gray.pixels, [20, 20, 20, 40, 40, 40]);
        let yuyv = decode_to_bgr(&capture("YUYV", vec![16, 128, 235, 128])).unwrap();
        assert_eq!(yuyv.pixels, [0, 0, 0, 255, 255, 255]);
        assert!(decode_to_bgr(&capture("YUYV", vec![0])).is_err());
        assert!(decode_to_bgr(&capture("GREY", vec![0])).is_err());
    }
    #[test]
    fn rgb_color_is_preserved_as_bgr() {
        let mut data = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut data, 100)
            .encode(
                &[255, 0, 0, 255, 0, 0],
                2,
                1,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        let bgr = decode_to_bgr(&capture("MJPG", data)).unwrap();
        assert!(bgr.pixels[2] > 240 && bgr.pixels[0] < 15 && bgr.pixels[1] < 15);
    }
}
