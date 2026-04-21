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
        let output = Command::new(&self.program)
            .args(&self.args)
            .output()
            .with_context(|| {
                format!("failed to run emitter command '{}'", self.program.display())
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            anyhow::bail!(
                "emitter command '{}' failed (exit={} stdout='{}' stderr='{}')",
                self.program.display(),
                output.status,
                stdout,
                stderr
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if stdout.is_empty() {
            Ok(())
        } else {
            eprintln!("emitter: ran '{}': {}", self.program.display(), stdout);
            Ok(())
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
