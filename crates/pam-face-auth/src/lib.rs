use std::ffi::{CStr, CString};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::ptr;
use std::time::Duration;

use common::protocol::{
    encode_request, AuthenticateRequest, AuthenticateResponse, Request, Response,
    DEFAULT_SOCKET_PATH, PROTOCOL_VERSION,
};
use libc::{c_char, c_int};
use pam_sys::{
    pam_conv, pam_get_item, pam_get_user, pam_handle_t, pam_message, pam_response,
    PAM_AUTH_ERR, PAM_CONV, PAM_CONV_ERR, PAM_RHOST, PAM_SERVICE, PAM_SERVICE_ERR,
    PAM_SUCCESS, PAM_SYSTEM_ERR, PAM_TEXT_INFO, PAM_TTY, PAM_USER_UNKNOWN,
};

type PamResultCode = i32;
type PamFlag = i32;

unsafe fn pam_get_username(pamh: *mut pam_handle_t) -> Result<String, PamResultCode> {
    let mut user_ptr: *const c_char = ptr::null();
    let status = pam_get_user(pamh, &mut user_ptr, ptr::null());
    if status != PAM_SUCCESS || user_ptr.is_null() {
        return Err(PAM_USER_UNKNOWN);
    }

    CStr::from_ptr(user_ptr)
        .to_str()
        .map(|s| s.to_owned())
        .map_err(|_| PAM_USER_UNKNOWN)
}

unsafe fn pam_get_optional_item(pamh: *mut pam_handle_t, item_type: c_int) -> Option<String> {
    let mut item_ptr: *const libc::c_void = ptr::null();
    let status = pam_get_item(pamh, item_type, &mut item_ptr);
    if status != PAM_SUCCESS || item_ptr.is_null() {
        return None;
    }

    let c_str = CStr::from_ptr(item_ptr.cast::<c_char>());
    c_str.to_str().ok().map(|s| s.to_owned())
}

unsafe fn pam_text_info(pamh: *mut pam_handle_t, message: &str) -> PamResultCode {
    let mut conv_item: *const libc::c_void = ptr::null();
    let status = pam_get_item(pamh, PAM_CONV, &mut conv_item);
    if status != PAM_SUCCESS || conv_item.is_null() {
        return PAM_SUCCESS;
    }

    let conv = &*(conv_item as *const pam_conv);
    let Some(conv_fn) = conv.conv else {
        return PAM_SUCCESS;
    };

    let c_message = match CString::new(message) {
        Ok(m) => m,
        Err(_) => return PAM_SYSTEM_ERR,
    };
    let msg = pam_message {
        msg_style: PAM_TEXT_INFO,
        msg: c_message.as_ptr(),
    };
    let mut msg_ptr: *const pam_message = &msg;
    let mut resp: *mut pam_response = ptr::null_mut();

    let conv_status = conv_fn(1, &mut msg_ptr, &mut resp, conv.appdata_ptr);
    if !resp.is_null() {
        // Some PAM conv implementations may allocate an empty response.
        if !(*resp).resp.is_null() {
            libc::free((*resp).resp.cast());
        }
        libc::free(resp.cast());
    }
    if conv_status == PAM_SUCCESS {
        PAM_SUCCESS
    } else {
        PAM_CONV_ERR
    }
}

fn daemon_authenticate(
    request: &AuthenticateRequest,
) -> Result<AuthenticateResponse, PamResultCode> {
    let mut stream = UnixStream::connect(DEFAULT_SOCKET_PATH).map_err(|_| PAM_SERVICE_ERR)?;
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .map_err(|_| PAM_SYSTEM_ERR)?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|_| PAM_SYSTEM_ERR)?;

    let payload = encode_request(&Request::Authenticate(AuthenticateRequest {
        version: request.version,
        username: request.username.clone(),
        service: request.service.clone(),
        tty: request.tty.clone(),
        rhost: request.rhost.clone(),
    }))
    .map_err(|_| PAM_SYSTEM_ERR)?;

    stream.write_all(&payload).map_err(|_| PAM_SERVICE_ERR)?;

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|_| PAM_SERVICE_ERR)?;

    match serde_json::from_str::<Response>(&line).map_err(|_| PAM_SYSTEM_ERR)? {
        Response::Authenticate(resp) => Ok(resp),
        Response::Error(_) => Err(PAM_AUTH_ERR),
        Response::Pong => Err(PAM_SYSTEM_ERR),
    }
}

#[no_mangle]
pub unsafe extern "C" fn pam_sm_setcred(
    _pamh: *mut pam_handle_t,
    _flags: PamFlag,
    _argc: c_int,
    _argv: *const *const c_char,
) -> PamResultCode {
    PAM_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn pam_sm_authenticate(
    pamh: *mut pam_handle_t,
    _flags: PamFlag,
    _argc: c_int,
    _argv: *const *const c_char,
) -> PamResultCode {
    let _ = pam_text_info(pamh, "Face authentication in progress. Look at the IR camera...");

    let username = match pam_get_username(pamh) {
        Ok(username) => username,
        Err(err) => return err,
    };

    let service = pam_get_optional_item(pamh, PAM_SERVICE);
    let tty = pam_get_optional_item(pamh, PAM_TTY);
    let rhost = pam_get_optional_item(pamh, PAM_RHOST);

    let request = AuthenticateRequest {
        version: PROTOCOL_VERSION,
        username,
        service,
        tty,
        rhost,
    };

    match daemon_authenticate(&request) {
        Ok(response) if response.success => PAM_SUCCESS,
        Ok(_) => PAM_AUTH_ERR,
        Err(code) => code,
    }
}

#[allow(dead_code)]
fn _c_string(input: &str) -> CString {
    CString::new(input).expect("CString::new failed")
}
