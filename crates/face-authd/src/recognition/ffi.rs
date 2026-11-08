use std::os::raw::{c_char, c_int};
pub enum FaceCtx {}
extern "C" {
    pub fn face_create(detector: *const c_char, recognizer: *const c_char) -> *mut FaceCtx;
    pub fn face_destroy(ctx: *mut FaceCtx);
    pub fn face_last_error() -> *const c_char;
    pub fn face_detect(
        ctx: *mut FaceCtx,
        pixels: *const u8,
        width: c_int,
        height: c_int,
        channels: c_int,
        rows: *mut f32,
        capacity: c_int,
    ) -> c_int;
    pub fn face_embed(
        ctx: *mut FaceCtx,
        pixels: *const u8,
        width: c_int,
        height: c_int,
        channels: c_int,
        face: *const f32,
        embedding: *mut f32,
    ) -> c_int;
}
