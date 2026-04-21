use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DEFAULT_SOCKET_PATH: &str = "/run/face-authd.sock";
pub const PROTOCOL_VERSION: u32 = 1;

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
