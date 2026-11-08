#include "face_backend.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect/face.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

static thread_local std::string error_message;
struct face_ctx {
    cv::Ptr<cv::FaceDetectorYN> detector;
    cv::Ptr<cv::FaceRecognizerSF> recognizer;
};
static cv::Mat bgr_image(const uint8_t* pixels, int width, int height, int channels) {
    if (!pixels || width <= 0 || height <= 0 || (channels != 1 && channels != 3))
        throw std::invalid_argument("invalid input image");
    cv::Mat view(height, width, channels == 1 ? CV_8UC1 : CV_8UC3, const_cast<uint8_t*>(pixels));
    if (channels == 3) return view;
    cv::Mat bgr; cv::cvtColor(view, bgr, cv::COLOR_GRAY2BGR); return bgr;
}
extern "C" {
face_ctx_t* face_create(const char* detector, const char* recognizer) {
    try {
        // Bounded CPU parallelism avoids oversubscription during local authentication.
        cv::setNumThreads(2);
        auto ctx = std::make_unique<face_ctx>();
        ctx->detector = cv::FaceDetectorYN::create(detector, "", cv::Size(320, 320),
            0.9f, 0.3f, 5000, cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);
        if (recognizer && recognizer[0])
            ctx->recognizer = cv::FaceRecognizerSF::create(recognizer, "", cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);
        return ctx.release();
    } catch (const std::exception& e) { error_message = e.what(); return nullptr; }
}
void face_destroy(face_ctx_t* ctx) { delete ctx; }
const char* face_last_error(void) { return error_message.c_str(); }
int face_detect(face_ctx_t* ctx, const uint8_t* pixels, int width, int height, int channels, float* rows, int capacity) {
    try {
        if (!ctx || !rows || capacity <= 0) throw std::invalid_argument("invalid detection context/output");
        auto image = bgr_image(pixels, width, height, channels);
        ctx->detector->setInputSize(image.size());
        cv::Mat faces; ctx->detector->detect(image, faces);
        if (faces.empty()) return 0;
        if (faces.type() != CV_32F || faces.cols != 15) throw std::runtime_error("unexpected YuNet output");
        const int count = std::min(faces.rows, capacity);
        for (int i = 0; i < count; ++i) std::memcpy(rows + i * 15, faces.ptr<float>(i), 15 * sizeof(float));
        return count;
    } catch (const std::exception& e) { error_message = e.what(); return -1; }
}
int face_embed(face_ctx_t* ctx, const uint8_t* pixels, int width, int height, int channels, const float* face, float* embedding) {
    try {
        if (!ctx || ctx->recognizer.empty() || !face || !embedding) throw std::invalid_argument("missing recognition context/output");
        auto image = bgr_image(pixels, width, height, channels);
        cv::Mat box(1, 15, CV_32F, const_cast<float*>(face));
        cv::Mat aligned, feature;
        ctx->recognizer->alignCrop(image, box, aligned);
        ctx->recognizer->feature(aligned, feature);
        if (feature.type() != CV_32F || feature.total() != 128) throw std::runtime_error("unexpected SFace feature shape");
        const double norm = cv::norm(feature, cv::NORM_L2);
        if (!std::isfinite(norm) || norm <= 1e-12) throw std::runtime_error("invalid SFace feature norm");
        cv::Mat unit; feature.convertTo(unit, CV_32F, 1.0 / norm);
        const float* values = unit.ptr<float>();
        for (int i = 0; i < 128; ++i) {
            if (!std::isfinite(values[i])) throw std::runtime_error("non-finite SFace feature");
            embedding[i] = values[i];
        }
        return 0;
    } catch (const std::exception& e) { error_message = e.what(); return -1; }
}
}
