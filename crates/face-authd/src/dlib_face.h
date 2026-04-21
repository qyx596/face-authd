#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct dlib_ctx dlib_ctx_t;

/* Load models and create a context.
   sp5_path  : path to shape_predictor_5_face_landmarks.dat (required)
   rec_path  : path to dlib_face_recognition_resnet_model_v1.dat
               Pass NULL to skip loading — dlib_embed() will return -1.
   Returns NULL on failure; call dlib_last_error() for details. */
dlib_ctx_t* dlib_create(const char* sp5_path, const char* rec_path);
void        dlib_destroy(dlib_ctx_t* ctx);
const char* dlib_last_error(void);

/* Detect faces in a grayscale image (row-major, 1 byte per pixel).
   out_rects : float[max_faces * 4]  — [x, y, w, h] per face
   out_scores: float[max_faces]      — detection confidence per face
   out_lms   : float[max_faces * 10] — 5 landmarks × [x, y] per face
   Returns the number of faces written, or -1 on error. */
int dlib_detect(dlib_ctx_t* ctx,
                const uint8_t* gray, int width, int height,
                float* out_rects, float* out_scores, float* out_lms,
                int max_faces);

/* Compute a 128-dim L2-normalised face descriptor.
   Requires rec_path to have been provided at dlib_create().
   Returns 0 on success, -1 on error. */
int dlib_embed(dlib_ctx_t* ctx,
               const uint8_t* gray, int width, int height,
               float bx, float by, float bw, float bh,
               float* out_emb);  /* 128 floats */

#ifdef __cplusplus
}
#endif
