#!/usr/bin/env bash
set -euo pipefail
# SHA-256 values are the pinned upstream Git LFS object IDs.
MODEL_REV=48b10fa505aeb51fbd127fc60a5c660040e823b7
MODEL_NAMES=(face_detection_yunet_2023mar.onnx face_recognition_sface_2021dec.onnx)
MODEL_FOLDERS=(face_detection_yunet face_recognition_sface)
MODEL_HASHES=(8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79)
model_cache="${1:?usage: download-models.sh CACHE_DIR}"
mkdir -p "$model_cache"
model_tmp="$(mktemp -d "$model_cache/.download.XXXXXX")"
trap 'rm -rf "$model_tmp"' EXIT
for i in "${!MODEL_NAMES[@]}"; do
    name="${MODEL_NAMES[$i]}"
    hash="${MODEL_HASHES[$i]}"
    if printf '%s  %s\n' "$hash" "$model_cache/$name" | sha256sum --check --status; then continue; fi
    curl --proto '=https' --proto-redir '=https' --location --fail --connect-timeout 10 --max-time 90 \
        --max-filesize 67108864 --output "$model_tmp/$name" \
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/$MODEL_REV/models/${MODEL_FOLDERS[$i]}/$name"
    printf '%s  %s\n' "$hash" "$model_tmp/$name" | sha256sum --check
    mv "$model_tmp/$name" "$model_cache/$name"
done
