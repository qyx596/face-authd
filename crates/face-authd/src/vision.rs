use crate::camera::GrayscaleFrame;

#[derive(Debug, Clone)]
pub struct FrameStats {
    pub mean_luma: f32,
    pub min_luma: u8,
    pub max_luma: u8,
    pub stddev_luma: f32,
    pub dark_pixel_ratio: f32,
    pub bright_pixel_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    pub usable_for_face_recognition: bool,
    pub stats: FrameStats,
    pub notes: Vec<String>,
}

pub trait FrameAnalyzer: Send + Sync {
    fn analyze(&self, frame: &GrayscaleFrame) -> FrameAnalysis;
}

#[derive(Default)]
pub struct StubFrameAnalyzer;

impl FrameAnalyzer for StubFrameAnalyzer {
    fn analyze(&self, frame: &GrayscaleFrame) -> FrameAnalysis {
        let stats = compute_frame_stats(frame);
        let mut notes = Vec::new();

        if stats.dark_pixel_ratio > 0.70 {
            notes.push(
                "frame is mostly dark; IR emitter may be inactive or subject too far".to_string(),
            );
        }

        if stats.bright_pixel_ratio > 0.30 {
            notes.push("frame contains a large number of saturated bright pixels".to_string());
        }

        if stats.stddev_luma < 12.0 {
            notes.push("frame contrast is very low".to_string());
        }

        if stats.mean_luma < 35.0 {
            notes.push("average luma is low".to_string());
        }

        let usable =
            stats.dark_pixel_ratio <= 0.70 && stats.stddev_luma >= 12.0 && stats.mean_luma >= 35.0;

        if usable {
            notes.push("frame quality is acceptable for the next detection stage".to_string());
        }

        FrameAnalysis {
            usable_for_face_recognition: usable,
            stats,
            notes,
        }
    }
}

pub fn format_analysis(analysis: &FrameAnalysis) -> String {
    format!(
        "usable={} mean={:.1} min={} max={} stddev={:.1} dark={:.1}% bright={:.1}% notes={}",
        analysis.usable_for_face_recognition,
        analysis.stats.mean_luma,
        analysis.stats.min_luma,
        analysis.stats.max_luma,
        analysis.stats.stddev_luma,
        analysis.stats.dark_pixel_ratio * 100.0,
        analysis.stats.bright_pixel_ratio * 100.0,
        if analysis.notes.is_empty() {
            "none".to_string()
        } else {
            analysis.notes.join(" | ")
        }
    )
}

fn compute_frame_stats(frame: &GrayscaleFrame) -> FrameStats {
    let total = frame.pixels.len().max(1) as f32;

    let mut sum = 0f32;
    let mut min_luma = u8::MAX;
    let mut max_luma = u8::MIN;
    let mut dark_pixels = 0usize;
    let mut bright_pixels = 0usize;

    for &pixel in &frame.pixels {
        sum += pixel as f32;
        min_luma = min_luma.min(pixel);
        max_luma = max_luma.max(pixel);

        if pixel <= 32 {
            dark_pixels += 1;
        }
        if pixel >= 224 {
            bright_pixels += 1;
        }
    }

    let mean = sum / total;
    let variance = frame
        .pixels
        .iter()
        .map(|&pixel| {
            let diff = pixel as f32 - mean;
            diff * diff
        })
        .sum::<f32>()
        / total;

    FrameStats {
        mean_luma: mean,
        min_luma,
        max_luma,
        stddev_luma: variance.sqrt(),
        dark_pixel_ratio: dark_pixels as f32 / total,
        bright_pixel_ratio: bright_pixels as f32 / total,
    }
}
