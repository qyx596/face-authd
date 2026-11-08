#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct face_ctx face_ctx_t;
face_ctx_t* face_create(const char* detector, const char* recognizer);
void face_destroy(face_ctx_t* ctx);
const char* face_last_error(void);
int face_detect(face_ctx_t*, const uint8_t*, int width, int height, int channels, float* rows, int capacity);
int face_embed(face_ctx_t*, const uint8_t*, int width, int height, int channels, const float* face, float* embedding);
#ifdef __cplusplus
}
#endif
