#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PKG_NAME="face-authd"
PKG_VERSION="${PKG_VERSION:-0.2.0}"
HOST_ARCH="$(dpkg --print-architecture)"
ARCH="${ARCH:-$HOST_ARCH}"
if [ "$ARCH" != "$HOST_ARCH" ]; then
    echo "error: this script builds native packages only; ARCH must match $HOST_ARCH" >&2
    exit 1
fi
PAM_MULTIARCH="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
BUILD_DIR="$ROOT_DIR/target/deb-build/${PKG_NAME}_${PKG_VERSION}_${ARCH}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$ROOT_DIR/target/deb-model-cache}"

DET_FILE="face_detection_yunet_2023mar.onnx"
REC_FILE="face_recognition_sface_2021dec.onnx"

echo "==> Building release binaries"
cargo build --locked --release -p pam-face-auth -p face-authd

# Resolve the actual runtime OpenCV package instead of omitting it or assuming a
# fixed SONAME/package version across Debian/Ubuntu releases.
OPENCV_PC=opencv4
if ! pkg-config --exists opencv4; then OPENCV_PC=opencv5; fi
OPENCV_LIBRARY="$(readlink -f "$(pkg-config --variable=libdir "$OPENCV_PC")/libopencv_objdetect.so")"
OPENCV_PACKAGE="$(dpkg-query -S "$OPENCV_LIBRARY" | head -n 1 | cut -d: -f1)"
if [ -z "$OPENCV_PACKAGE" ]; then
    echo "error: OpenCV must be installed from a Debian package" >&2
    exit 1
fi

echo "==> Preparing package layout at $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/usr/local/bin"
mkdir -p "$BUILD_DIR/lib/$PAM_MULTIARCH/security"
mkdir -p "$BUILD_DIR/etc/systemd/system"
mkdir -p "$BUILD_DIR/var/lib/face-authd/models"
mkdir -p "$BUILD_DIR/usr/share/doc/$PKG_NAME"
mkdir -p "$BUILD_DIR/usr/share/$PKG_NAME/examples"

install -m 0755 target/release/face-authd "$BUILD_DIR/usr/local/bin/face-authd"
install -m 0755 target/release/libpam_face_auth.so \
  "$BUILD_DIR/lib/$PAM_MULTIARCH/security/pam_face_auth.so"
install -m 0644 systemd/face-authd.service "$BUILD_DIR/etc/systemd/system/face-authd.service"
install -m 0644 README.md "$BUILD_DIR/usr/share/doc/$PKG_NAME/README.md"
install -m 0644 third_party/YuNet-LICENSE third_party/SFace-LICENSE third_party/NOTICE "$BUILD_DIR/usr/share/doc/$PKG_NAME/"
install -m 0644 examples/pam-sudo "$BUILD_DIR/usr/share/$PKG_NAME/examples/pam-sudo"

bash scripts/download-models.sh "$MODEL_CACHE_DIR"
install -m 0644 "$MODEL_CACHE_DIR/$DET_FILE" "$BUILD_DIR/var/lib/face-authd/models/$DET_FILE"
install -m 0644 "$MODEL_CACHE_DIR/$REC_FILE" "$BUILD_DIR/var/lib/face-authd/models/$REC_FILE"

cat >"$BUILD_DIR/DEBIAN/control" <<EOF
Package: $PKG_NAME
Version: $PKG_VERSION
Section: admin
Priority: optional
Architecture: $ARCH
Maintainer: YuxuanQiu <yuxuanqiu596@gmail.com>
Depends: $OPENCV_PACKAGE, libc6, libgcc-s1, libstdc++6, libpam0g, libsqlite3-0, libblas3, liblapack3, systemd, debconf (>= 0.5) | debconf-2.0
Description: Local face authentication daemon and PAM module
 A local face authentication stack for Linux:
 daemon (face-authd) + PAM module (pam_face_auth.so).
EOF

install -m 0755 packaging/deb/config "$BUILD_DIR/DEBIAN/config"
install -m 0755 packaging/deb/postinst "$BUILD_DIR/DEBIAN/postinst"
install -m 0755 packaging/deb/prerm "$BUILD_DIR/DEBIAN/prerm"
install -m 0755 packaging/deb/postrm "$BUILD_DIR/DEBIAN/postrm"
install -m 0644 packaging/deb/templates "$BUILD_DIR/DEBIAN/templates"

mkdir -p "$OUT_DIR"
DEB_FILE="$OUT_DIR/${PKG_NAME}_${PKG_VERSION}_${ARCH}.deb"

echo "==> Building deb package"
dpkg-deb --root-owner-group --build "$BUILD_DIR" "$DEB_FILE"

echo
echo "Built package:"
echo "  $DEB_FILE"
echo
echo "Install with:"
echo "  sudo dpkg -i $DEB_FILE"
