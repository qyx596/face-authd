use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub enum EmitterMode {
    None,
    ExternalCommand(ExternalEmitterCommand),
}

#[derive(Debug, Clone)]
pub struct ExternalEmitterCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
}

pub trait EmitterController: Send + Sync {
    fn activate(&self, device: Option<&Path>) -> Result<()>;
}

impl EmitterController for EmitterMode {
    fn activate(&self, device: Option<&Path>) -> Result<()> {
        match self {
            EmitterMode::None => Ok(()),
            EmitterMode::ExternalCommand(command) => command.activate(device),
        }
    }
}

impl EmitterController for ExternalEmitterCommand {
    fn activate(&self, _device: Option<&Path>) -> Result<()> {
        // Bound external hardware helpers; never block a Tokio runtime thread.
        let mut child = Command::new(&self.program)
            .args(&self.args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .with_context(|| {
                format!("failed to run emitter command '{}'", self.program.display())
            })?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    anyhow::ensure!(status.success(), "emitter command failed: {status}");
                    return Ok(());
                }
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10))
                }
                result => {
                    let _ = child.kill();
                    let _ = child.wait();
                    result.context("failed to wait for emitter command")?;
                    anyhow::bail!("emitter command timed out");
                }
            }
        }
    }
}

pub fn emitter_from_env() -> EmitterMode {
    match std::env::var("FACE_AUTHD_EMITTER_MODE").ok().as_deref() {
        Some("external-command") => {
            let program = std::env::var_os("FACE_AUTHD_EMITTER_CMD")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("linux-enable-ir-emitter"));
            let args = std::env::var("FACE_AUTHD_EMITTER_ARGS")
                .ok()
                .map(|value| value.split_whitespace().map(ToString::to_string).collect())
                .filter(|args: &Vec<String>| !args.is_empty())
                .unwrap_or_else(|| vec!["run".to_string()]);
            EmitterMode::ExternalCommand(ExternalEmitterCommand { program, args })
        }
        _ => EmitterMode::None,
    }
}
