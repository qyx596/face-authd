use std::io;
use std::mem;
use std::os::raw::c_void;
use std::path::Path;

use anyhow::{Context, Result};
use v4l::v4l2;
use v4l::v4l_sys::{
    v4l2_control, v4l2_query_ext_ctrl, V4L2_CTRL_FLAG_DISABLED, V4L2_CTRL_FLAG_GRABBED,
    V4L2_CTRL_FLAG_INACTIVE, V4L2_CTRL_FLAG_NEXT_COMPOUND, V4L2_CTRL_FLAG_NEXT_CTRL,
    V4L2_CTRL_FLAG_READ_ONLY, V4L2_CTRL_FLAG_SLIDER, V4L2_CTRL_FLAG_UPDATE,
    V4L2_CTRL_FLAG_VOLATILE, V4L2_CTRL_FLAG_WRITE_ONLY,
};

const CTRL_TYPE_INTEGER: u32 = 1;
const CTRL_TYPE_BOOLEAN: u32 = 2;
const CTRL_TYPE_MENU: u32 = 3;
const CTRL_TYPE_BUTTON: u32 = 4;
const CTRL_TYPE_INTEGER64: u32 = 5;
const CTRL_TYPE_CTRL_CLASS: u32 = 6;
const CTRL_TYPE_STRING: u32 = 7;
const CTRL_TYPE_BITMASK: u32 = 8;
const CTRL_TYPE_INTEGER_MENU: u32 = 9;

#[derive(Debug, Clone)]
pub struct RawDeviceControl {
    pub id: u32,
    pub name: String,
    pub typ: u32,
    pub typ_name: String,
    pub minimum: i64,
    pub maximum: i64,
    pub step: u64,
    pub default: i64,
    pub flags_text: String,
    pub current: Option<String>,
}

pub fn query_device_controls(device_path: &Path) -> Result<Vec<RawDeviceControl>> {
    let fd = open_device(device_path)?;
    let result = unsafe { query_controls_from_fd(fd) };
    close_device(fd);
    result
}

pub fn set_device_control(device_path: &Path, control_name: &str, raw_value: &str) -> Result<()> {
    let fd = open_device(device_path)?;
    let result = unsafe { set_control_on_fd(fd, control_name, raw_value) };
    close_device(fd);
    result
}

fn open_device(device_path: &Path) -> Result<i32> {
    let path = device_path
        .to_str()
        .with_context(|| format!("device path is not valid UTF-8: {}", device_path.display()))?;
    v4l2::open(path, libc::O_RDWR)
        .with_context(|| format!("failed to open video device {}", device_path.display()))
}

fn close_device(fd: i32) {
    unsafe {
        libc::close(fd);
    }
}

unsafe fn query_controls_from_fd(fd: i32) -> Result<Vec<RawDeviceControl>> {
    let mut controls = Vec::new();
    let mut query: v4l2_query_ext_ctrl = mem::zeroed();

    loop {
        query.id |= V4L2_CTRL_FLAG_NEXT_CTRL | V4L2_CTRL_FLAG_NEXT_COMPOUND;
        match v4l2::ioctl(
            fd,
            v4l2::vidioc::VIDIOC_QUERY_EXT_CTRL,
            &mut query as *mut _ as *mut c_void,
        ) {
            Ok(()) => {
                let current = read_control_value(fd, query.id, query.type_);
                controls.push(RawDeviceControl {
                    id: query.id,
                    name: control_name(&query),
                    typ: query.type_,
                    typ_name: control_type_name(query.type_),
                    minimum: query.minimum,
                    maximum: query.maximum,
                    step: query.step,
                    default: query.default_value,
                    flags_text: control_flags_text(query.flags),
                    current,
                });
            }
            Err(err) => {
                if controls.is_empty() || err.kind() != io::ErrorKind::InvalidInput {
                    return Err(err).context("raw V4L2 control enumeration failed");
                }
                break;
            }
        }
    }

    Ok(controls)
}

unsafe fn set_control_on_fd(fd: i32, control_name: &str, raw_value: &str) -> Result<()> {
    let controls = query_controls_from_fd(fd)?;
    let control = controls
        .iter()
        .find(|control| {
            control.name.eq_ignore_ascii_case(control_name)
                || normalize_control_name(&control.name) == normalize_control_name(control_name)
        })
        .with_context(|| format!("control '{}' was not found", control_name))?;

    let mut v4l_ctrl: v4l2_control = mem::zeroed();
    v4l_ctrl.id = control.id;
    v4l_ctrl.value = parse_settable_value(control, raw_value)? as i32;

    v4l2::ioctl(
        fd,
        v4l2::vidioc::VIDIOC_S_CTRL,
        &mut v4l_ctrl as *mut _ as *mut c_void,
    )
    .with_context(|| format!("failed to set control '{}'", control.name))?;

    Ok(())
}

unsafe fn read_control_value(fd: i32, id: u32, typ: u32) -> Option<String> {
    if !matches!(
        typ,
        CTRL_TYPE_INTEGER
            | CTRL_TYPE_BOOLEAN
            | CTRL_TYPE_MENU
            | CTRL_TYPE_BITMASK
            | CTRL_TYPE_BUTTON
    ) {
        return None;
    }

    let mut ctrl: v4l2_control = mem::zeroed();
    ctrl.id = id;

    if v4l2::ioctl(
        fd,
        v4l2::vidioc::VIDIOC_G_CTRL,
        &mut ctrl as *mut _ as *mut c_void,
    )
    .is_err()
    {
        return None;
    }

    Some(match typ {
        CTRL_TYPE_BOOLEAN => (ctrl.value != 0).to_string(),
        _ => ctrl.value.to_string(),
    })
}

fn parse_settable_value(control: &RawDeviceControl, raw_value: &str) -> Result<i64> {
    match control.typ {
        CTRL_TYPE_BOOLEAN => match raw_value.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(1),
            "0" | "false" | "no" | "off" => Ok(0),
            _ => anyhow::bail!("invalid boolean value '{}'", raw_value),
        },
        CTRL_TYPE_INTEGER | CTRL_TYPE_MENU | CTRL_TYPE_BITMASK | CTRL_TYPE_BUTTON => raw_value
            .parse::<i64>()
            .with_context(|| format!("invalid integer value '{}'", raw_value)),
        CTRL_TYPE_INTEGER64 => anyhow::bail!(
            "control '{}' is INTEGER64; raw setter currently supports only values representable through VIDIOC_S_CTRL",
            control.name
        ),
        _ => anyhow::bail!(
            "control '{}' has unsupported settable type {}",
            control.name,
            control.typ_name
        ),
    }
}

fn control_name(query: &v4l2_query_ext_ctrl) -> String {
    let bytes = query.name.map(|byte| byte as u8);
    let nul = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..nul]).trim().to_string()
}

fn control_type_name(typ: u32) -> String {
    match typ {
        CTRL_TYPE_INTEGER => "Integer",
        CTRL_TYPE_BOOLEAN => "Boolean",
        CTRL_TYPE_MENU => "Menu",
        CTRL_TYPE_BUTTON => "Button",
        CTRL_TYPE_INTEGER64 => "Integer64",
        CTRL_TYPE_CTRL_CLASS => "CtrlClass",
        CTRL_TYPE_STRING => "String",
        CTRL_TYPE_BITMASK => "Bitmask",
        CTRL_TYPE_INTEGER_MENU => "IntegerMenu",
        _ => "Unknown",
    }
    .to_string()
}

fn control_flags_text(flags: u32) -> String {
    let mut parts = Vec::new();
    if flags & V4L2_CTRL_FLAG_DISABLED != 0 {
        parts.push("disabled");
    }
    if flags & V4L2_CTRL_FLAG_GRABBED != 0 {
        parts.push("grabbed");
    }
    if flags & V4L2_CTRL_FLAG_READ_ONLY != 0 {
        parts.push("read-only");
    }
    if flags & V4L2_CTRL_FLAG_UPDATE != 0 {
        parts.push("update");
    }
    if flags & V4L2_CTRL_FLAG_INACTIVE != 0 {
        parts.push("inactive");
    }
    if flags & V4L2_CTRL_FLAG_SLIDER != 0 {
        parts.push("slider");
    }
    if flags & V4L2_CTRL_FLAG_WRITE_ONLY != 0 {
        parts.push("write-only");
    }
    if flags & V4L2_CTRL_FLAG_VOLATILE != 0 {
        parts.push("volatile");
    }

    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("|")
    }
}

fn normalize_control_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}
