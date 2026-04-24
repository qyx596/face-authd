#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PKG_NAME="face-authd"
PKG_VERSION="${PKG_VERSION:-0.1.2}"
ARCH="${ARCH:-$(dpkg --print-architecture)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist}"
BUILD_DIR="$ROOT_DIR/target/deb-build/${PKG_NAME}_${PKG_VERSION}_${ARCH}"
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

echo "==> Preparing package layout at $BUILD_DIR"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/usr/local/bin"
mkdir -p "$BUILD_DIR/lib/x86_64-linux-gnu/security"
mkdir -p "$BUILD_DIR/etc/systemd/system"
mkdir -p "$BUILD_DIR/var/lib/face-authd/models"
mkdir -p "$BUILD_DIR/usr/share/doc/$PKG_NAME"
mkdir -p "$BUILD_DIR/usr/share/$PKG_NAME/examples"

install -m 0755 target/release/face-authd "$BUILD_DIR/usr/local/bin/face-authd"
install -m 0755 target/release/libpam_face_auth.so \
  "$BUILD_DIR/lib/x86_64-linux-gnu/security/pam_face_auth.so"
install -m 0644 systemd/face-authd.service "$BUILD_DIR/etc/systemd/system/face-authd.service"
install -m 0644 README.md "$BUILD_DIR/usr/share/doc/$PKG_NAME/README.md"
install -m 0644 examples/pam-sudo "$BUILD_DIR/usr/share/$PKG_NAME/examples/pam-sudo"

download_and_unpack_model "$SP5_URL" "$SP5_FILE"
download_and_unpack_model "$REC_URL" "$REC_FILE"
install -m 0644 "$MODEL_CACHE_DIR/$SP5_FILE" "$BUILD_DIR/var/lib/face-authd/models/$SP5_FILE"
install -m 0644 "$MODEL_CACHE_DIR/$REC_FILE" "$BUILD_DIR/var/lib/face-authd/models/$REC_FILE"

cat >"$BUILD_DIR/DEBIAN/control" <<EOF
Package: $PKG_NAME
Version: $PKG_VERSION
Section: admin
Priority: optional
Architecture: $ARCH
Maintainer: YuxuanQiu <yuxuanqiu596@gmail.com>
Depends: libc6, libgcc-s1, libstdc++6, libpam0g, libsqlite3-0, libblas3, liblapack3, systemd, debconf (>= 0.5) | debconf-2.0
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
