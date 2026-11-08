#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v fpm >/dev/null 2>&1; then
    echo "error: fpm is required. Install with: gem install --user-install fpm" >&2
    exit 1
fi

PKG_NAME="face-authd"
PKG_VERSION="${PKG_VERSION:-0.2.0}"
HOST_ARCH="$(uname -m)"
RPM_ARCH="${RPM_ARCH:-$HOST_ARCH}"
if [ "$RPM_ARCH" != "$HOST_ARCH" ]; then
  echo "error: this script builds native packages only" >&2
  exit 1
fi
PAM_LIBDIR="$(rpm --eval '%{_libdir}')"
OPENCV_PC=opencv4
if ! pkg-config --exists opencv4; then OPENCV_PC=opencv5; fi
OPENCV_LIBRARY="$(readlink -f "$(pkg-config --variable=libdir "$OPENCV_PC")/libopencv_objdetect.so")"
OPENCV_PACKAGE="$(rpm -qf --qf '%{NAME}' "$OPENCV_LIBRARY")"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
STAGE_ROOT="$ROOT_DIR/target/rpm-build/root"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$ROOT_DIR/target/deb-model-cache}"

DET_FILE="face_detection_yunet_2023mar.onnx"
REC_FILE="face_recognition_sface_2021dec.onnx"

echo "==> Building release binaries"
cargo build --locked --release -p pam-face-auth -p face-authd

echo "==> Preparing RPM staging tree at $STAGE_ROOT"
rm -rf "$STAGE_ROOT"
mkdir -p "$STAGE_ROOT/usr/local/bin"
mkdir -p "$STAGE_ROOT$PAM_LIBDIR/security"
mkdir -p "$STAGE_ROOT/etc/systemd/system"
mkdir -p "$STAGE_ROOT/var/lib/face-authd/models"
mkdir -p "$STAGE_ROOT/usr/share/doc/$PKG_NAME"

install -m 0755 target/release/face-authd "$STAGE_ROOT/usr/local/bin/face-authd"
install -m 0755 target/release/libpam_face_auth.so "$STAGE_ROOT$PAM_LIBDIR/security/pam_face_auth.so"
install -m 0644 systemd/face-authd.service "$STAGE_ROOT/etc/systemd/system/face-authd.service"
install -m 0644 README.md "$STAGE_ROOT/usr/share/doc/$PKG_NAME/README.md"
install -m 0644 third_party/YuNet-LICENSE third_party/SFace-LICENSE third_party/NOTICE "$STAGE_ROOT/usr/share/doc/$PKG_NAME/"

bash scripts/download-models.sh "$MODEL_CACHE_DIR"
install -m 0644 "$MODEL_CACHE_DIR/$DET_FILE" "$STAGE_ROOT/var/lib/face-authd/models/$DET_FILE"
install -m 0644 "$MODEL_CACHE_DIR/$REC_FILE" "$STAGE_ROOT/var/lib/face-authd/models/$REC_FILE"

mkdir -p "$OUT_DIR"

echo "==> Building RPM package"
fpm -s dir -t rpm \
  -n "$PKG_NAME" \
  -v "$PKG_VERSION" \
  --architecture "$RPM_ARCH" \
  --description "Local face authentication daemon and PAM module" \
  --depends "$OPENCV_PACKAGE" \
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
