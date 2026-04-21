#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v fpm >/dev/null 2>&1; then
    echo "error: fpm is required. Install with: gem install --user-install fpm" >&2
    exit 1
fi

PKG_NAME="face-authd"
PKG_VERSION="${PKG_VERSION:-0.1.0}"
RPM_ARCH="${RPM_ARCH:-$(uname -m)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
STAGE_ROOT="$ROOT_DIR/target/rpm-build/root"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$ROOT_DIR/target/deb-model-cache}"

SP5_URL="https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2"
REC_URL="http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
SP5_FILE="shape_predictor_5_face_landmarks.dat"
REC_FILE="dlib_face_recognition_resnet_model_v1.dat"

download_and_unpack_model() {
    local url="$1"
    local out_file="$2"
    local bz2_path="$MODEL_CACHE_DIR/${out_file}.bz2"
    local dat_path="$MODEL_CACHE_DIR/${out_file}"

    mkdir -p "$MODEL_CACHE_DIR"
    if [ ! -f "$dat_path" ]; then
        echo "==> Downloading model: $out_file"
        curl -L --fail -o "$bz2_path" "$url"
        bzip2 -dc "$bz2_path" > "$dat_path"
    fi
}

echo "==> Building release binaries"
cargo build --release -p pam-face-auth -p face-authd

echo "==> Preparing RPM staging tree at $STAGE_ROOT"
rm -rf "$STAGE_ROOT"
mkdir -p "$STAGE_ROOT/usr/local/bin"
mkdir -p "$STAGE_ROOT/usr/lib64/security"
mkdir -p "$STAGE_ROOT/etc/systemd/system"
mkdir -p "$STAGE_ROOT/var/lib/face-authd/models"
mkdir -p "$STAGE_ROOT/usr/share/doc/$PKG_NAME"

install -m 0755 target/release/face-authd "$STAGE_ROOT/usr/local/bin/face-authd"
install -m 0755 target/release/libpam_face_auth.so "$STAGE_ROOT/usr/lib64/security/pam_face_auth.so"
install -m 0644 systemd/face-authd.service "$STAGE_ROOT/etc/systemd/system/face-authd.service"
install -m 0644 README.md "$STAGE_ROOT/usr/share/doc/$PKG_NAME/README.md"

download_and_unpack_model "$SP5_URL" "$SP5_FILE"
download_and_unpack_model "$REC_URL" "$REC_FILE"
install -m 0644 "$MODEL_CACHE_DIR/$SP5_FILE" "$STAGE_ROOT/var/lib/face-authd/models/$SP5_FILE"
install -m 0644 "$MODEL_CACHE_DIR/$REC_FILE" "$STAGE_ROOT/var/lib/face-authd/models/$REC_FILE"

mkdir -p "$OUT_DIR"

echo "==> Building RPM package"
fpm -s dir -t rpm \
  -n "$PKG_NAME" \
  -v "$PKG_VERSION" \
  --architecture "$RPM_ARCH" \
  --description "Local face authentication daemon and PAM module" \
  --depends "pam" \
  --depends "systemd" \
  --depends "sqlite" \
  --depends "blas" \
  --depends "lapack" \
  --after-install packaging/rpm/postinstall.sh \
  --before-remove packaging/rpm/preremove.sh \
  --after-remove packaging/rpm/postremove.sh \
  -C "$STAGE_ROOT" \
  --package "$OUT_DIR"

echo
echo "Built RPM package(s):"
ls -1 "$OUT_DIR"/*.rpm
