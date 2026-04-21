use std::os::raw::{c_char, c_int};

pub enum DlibCtx {}

extern "C" {
    pub fn dlib_create(sp5_path: *const c_char, rec_path: *const c_char) -> *mut DlibCtx;
    pub fn dlib_destroy(ctx: *mut DlibCtx);
    pub fn dlib_last_error() -> *const c_char;

    pub fn dlib_detect(
        ctx: *mut DlibCtx,
        gray: *const u8,
        width: c_int,
        height: c_int,
        out_rects: *mut f32,
        out_scores: *mut f32,
        out_lms: *mut f32,
        max_faces: c_int,
    ) -> c_int;

    pub fn dlib_embed(
        ctx: *mut DlibCtx,
        gray: *const u8,
        width: c_int,
        height: c_int,
        bx: f32,
        by: f32,
        bw: f32,
        bh: f32,
        out_emb: *mut f32,
    ) -> c_int;
}
