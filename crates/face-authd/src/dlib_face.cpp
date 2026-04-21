// dlib face detection and recognition — C wrapper for Rust FFI.
//
// Detection : dlib frontal HOG detector + 5-point shape predictor
// Recognition: dlib metric-learning model (128-dim descriptor)
//              same architecture as dlib_face_recognition_resnet_model_v1.dat

#include "dlib_face.h"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/dnn.h>

#include <string>
#include <vector>
#include <algorithm>

using namespace dlib;

// ---------------------------------------------------------------------------
// Network architecture — must match the serialised weights exactly.
// Copied from dlib/examples/dnn_face_recognition_ex.cpp
// ---------------------------------------------------------------------------

template <template <int,template<typename>class,int,typename> class block,
          int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block,
          int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block_t = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block_t,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block_t,N,affine,SUBNET>>;

template <typename SUBNET> using level0 = ares_down<256,SUBNET>;
template <typename SUBNET> using level1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using level2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using level3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using level4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
    level0<level1<level2<level3<level4<
    max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

static thread_local std::string g_last_error;

struct dlib_ctx {
    frontal_face_detector detector;
    shape_predictor       sp;
    anet_type             net;
    bool                  has_rec = false;
};

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

extern "C" {

dlib_ctx_t* dlib_create(const char* sp5_path, const char* rec_path)
{
    try {
        auto* ctx = new dlib_ctx;
        ctx->detector = get_frontal_face_detector();
        deserialize(sp5_path) >> ctx->sp;
        if (rec_path && rec_path[0] != '\0') {
            deserialize(rec_path) >> ctx->net;
            ctx->has_rec = true;
        }
        return ctx;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return nullptr;
    }
}

void dlib_destroy(dlib_ctx_t* ctx)
{
    delete ctx;
}

const char* dlib_last_error(void)
{
    return g_last_error.c_str();
}

int dlib_detect(dlib_ctx_t* ctx,
                const uint8_t* gray, int width, int height,
                float* out_rects, float* out_scores, float* out_lms,
                int max_faces)
{
    try {
        array2d<unsigned char> img(height, width);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                img[y][x] = gray[y * width + x];

        // operator()(img, rect_detections, threshold) provides confidence scores
        std::vector<rect_detection> dets;
        ctx->detector(img, dets, 0.5);

        // Sort by confidence descending
        std::sort(dets.begin(), dets.end(),
                  [](const rect_detection& a, const rect_detection& b){
                      return a.detection_confidence > b.detection_confidence;
                  });

        int n = static_cast<int>(std::min(dets.size(), (size_t)max_faces));
        for (int i = 0; i < n; ++i) {
            const rectangle& r = dets[i].rect;
            out_rects[i*4+0] = static_cast<float>(r.left());
            out_rects[i*4+1] = static_cast<float>(r.top());
            out_rects[i*4+2] = static_cast<float>(r.width());
            out_rects[i*4+3] = static_cast<float>(r.height());
            out_scores[i]    = static_cast<float>(dets[i].detection_confidence);

            full_object_detection shape = ctx->sp(img, r);
            for (int k = 0; k < 5; ++k) {
                out_lms[i*10 + k*2 + 0] = static_cast<float>(shape.part(k).x());
                out_lms[i*10 + k*2 + 1] = static_cast<float>(shape.part(k).y());
            }
        }
        return n;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -1;
    }
}

int dlib_embed(dlib_ctx_t* ctx,
               const uint8_t* gray, int width, int height,
               float bx, float by, float bw, float bh,
               float* out_emb)
{
    if (!ctx->has_rec) {
        g_last_error = "recognition model not loaded (rec_path was NULL at dlib_create)";
        return -1;
    }
    try {
        // Shape predictor works on grayscale
        array2d<unsigned char> gray_img(height, width);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                gray_img[y][x] = gray[y * width + x];

        // Recognition network expects RGB pixel type
        matrix<rgb_pixel> rgb_img(height, width);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                uint8_t v = gray[y * width + x];
                rgb_img(y, x) = rgb_pixel(v, v, v);
            }

        rectangle rect(
            static_cast<long>(bx),
            static_cast<long>(by),
            static_cast<long>(bx + bw),
            static_cast<long>(by + bh)
        );

        full_object_detection shape = ctx->sp(gray_img, rect);

        matrix<rgb_pixel> chip;
        extract_image_chip(rgb_img,
                           get_face_chip_details(shape, 150, 0.25),
                           chip);

        std::vector<matrix<rgb_pixel>> batch = { chip };
        std::vector<matrix<float,0,1>> descs = ctx->net(batch);

        const auto& d = descs[0];
        for (int i = 0; i < 128; ++i)
            out_emb[i] = d(i);

        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
        return -1;
    }
}

} // extern "C"
