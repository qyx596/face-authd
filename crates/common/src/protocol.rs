use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DEFAULT_SOCKET_PATH: &str = "/run/face-authd.sock";
pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_MESSAGE_BYTES: usize = 8192;

pub fn valid_username(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 256
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.' | b'$'))
}

impl AuthenticateRequest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != PROTOCOL_VERSION {
            return Err("protocol version mismatch");
        }
        if !valid_username(&self.username) {
            return Err("invalid username");
        }
        if self.rhost.as_deref().is_some_and(|s| !s.is_empty()) {
            return Err("remote face authentication is disabled");
        }
        if !matches!(
            self.service.as_deref(),
            Some("sudo" | "sudo-i" | "gdm-password" | "login" | "sddm" | "kde" | "polkit-1")
        ) {
            return Err("PAM service is not allowed");
        }
        if self
            .tty
            .as_ref()
            .is_some_and(|s| s.len() > 1024 || s.contains('\0'))
        {
            return Err("invalid tty");
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Authenticate(AuthenticateRequest),
    Ping,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthenticateRequest {
    pub version: u32,
    pub username: String,
    pub service: Option<String>,
    pub tty: Option<String>,
    pub rhost: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Authenticate(AuthenticateResponse),
    Pong,
    Error(ErrorResponse),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthenticateResponse {
    pub success: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub fn encode_request(request: &Request) -> Result<Vec<u8>, ProtocolError> {
    let mut payload = serde_json::to_vec(request)?;
    payload.push(b'\n');
    Ok(payload)
}

pub fn encode_response(response: &Response) -> Result<Vec<u8>, ProtocolError> {
    let mut payload = serde_json::to_vec(response)?;
    payload.push(b'\n');
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> AuthenticateRequest {
        AuthenticateRequest {
            version: PROTOCOL_VERSION,
            username: "alice".into(),
            service: Some("sudo".into()),
            tty: None,
            rhost: None,
        }
    }
    #[test]
    fn only_local_allowed_services() {
        let mut r = request();
        assert!(r.validate().is_ok());
        r.rhost = Some("localhost".into());
        assert!(r.validate().is_err());
        r.rhost = None;
        r.service = Some("sshd".into());
        assert!(r.validate().is_err());
        r.service = None;
        assert!(r.validate().is_err());
    }
    #[test]
    fn usernames_cannot_escape_or_collide() {
        for name in ["", ".", "..", "a/b", "a b", "a\0b"] {
            assert!(!valid_username(name));
        }
        for name in ["alice", "domain-user", "user.name", "machine$"] {
            assert!(valid_username(name));
        }
    }
}
