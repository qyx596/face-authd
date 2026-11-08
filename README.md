# face-authd

Local Linux face authentication through a Rust daemon and PAM module.
The default backend is **YuNet + SFace**, running locally on the CPU through
OpenCV. Images and templates are not sent to an external recognition service.
The project is experimental and does not implement validated liveness detection.

## Recognition pipeline

1. Negotiate a 640×360 capture stream (the driver may return another supported size).
2. Preserve color from MJPEG/YUYV cameras as BGR; repeat each IR grayscale value
   across B,G,R. Enrollment and authentication use the same preprocessing.
3. Detect faces with YuNet (`face_detection_yunet_2023mar.onnx`, confidence 0.9).
   Zero or multiple detected faces never produce an authentication descriptor.
4. Use YuNet's five landmarks and OpenCV SFace `alignCrop`, then extract a
   128-dimensional SFace feature and explicitly normalize it to unit length.
5. Compare normalized Euclidean distance. The initial maximum distance is
   **1.128**, taken from OpenCV's example, with a strict `<` comparison and two
   consecutive matching frames. Lower thresholds are stricter.

The threshold is a starting point from a visible-light benchmark, not a measured
false-accept guarantee for an IR sensor. A camera reopen/same-person smoke test
does not establish independent-session accuracy, impostor acceptance or resistance
to photos, screen replay or masks. IR pixel formats alone do not prove liveness.

## Models and licenses

Models are pinned to OpenCV Zoo revision
`48b10fa505aeb51fbd127fc60a5c660040e823b7` and verified by SHA-256 before loading.
The checksums match the upstream Git LFS object IDs. Downloads use HTTPS, bounded
sizes and timeouts, and atomic replacement. Authentication never downloads weights.

| Model | SHA-256 | License |
| --- | --- | --- |
| YuNet 2023mar | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT |
| SFace 2021dec | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

See `third_party/NOTICE`, `third_party/YuNet-LICENSE`, and
`third_party/SFace-LICENSE` when redistributing weights. The repository's MIT
license does not replace third-party licenses.

References: [OpenCV tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html),
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
[SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).

## Build

Install a Rust toolchain, C/C++17 compiler, pkg-config, clang (for V4L bindings),
PAM development headers, keyutils development files, OpenSSL development files,
and **OpenCV ≥4.8 or OpenCV 5** development files. Required OpenCV modules are
`core`, `imgproc`, `dnn`, and `objdetect`. GUI preview also needs normal X11/Wayland
build dependencies. dlib, BLAS/LAPACK and the old `.dat` files are no longer used.

OpenCV is discovered using `pkg-config opencv4` or `opencv5`. Some older distro
packages (including Ubuntu 24.04's OpenCV 4.6) are too old for the selected YuNet
architecture. Use a newer supported OpenCV build; CI builds pinned OpenCV 4.10.
A custom installation can be selected with `PKG_CONFIG_PATH` and, when necessary,
`LD_LIBRARY_PATH`.

```bash
cargo build --release --workspace --locked
sudo target/release/face-authd fetch-models
```

For a local evaluation without changing system state:

```bash
FACE_AUTHD_MODEL_DIR="$PWD/target/eval-models" target/release/face-authd fetch-models
FACE_AUTHD_MODEL_DIR="$PWD/target/eval-models" \
  target/release/face-authd benchmark --device /dev/video2 --frames 40 --json target/ir-result.json
```

`benchmark` provides a countdown, collects five in-memory reference samples,
closes/reopens the camera, and measures a verification window. If it cannot collect
five references it reports the partial/empty count rather than claiming successful
enrollment. No frames or embeddings are saved. JSON includes detection counts,
normalized match distances, first-confirmed-match time, inference/decode p50/p95,
actual format/resolution, average pixel value, and limitations. Use a release build
for timing. The measurement is a same-session smoke test, not an accuracy benchmark.

## Install and enroll

Install the daemon, service and PAM module in the paths appropriate to your distro:

```bash
sudo install -Dm755 target/release/face-authd /usr/local/bin/face-authd
sudo install -Dm644 systemd/face-authd.service /etc/systemd/system/face-authd.service
# Debian-style example (replace the multiarch directory for your platform):
sudo install -Dm755 target/release/libpam_face_auth.so /lib/x86_64-linux-gnu/security/pam_face_auth.so
# Fedora-style typical module directory: /usr/lib64/security/
sudo systemctl daemon-reload
sudo face-authd setup --user "$USER" --device /dev/video2
sudo systemctl enable --now face-authd
```

Only after verification should `pam_face_auth.so` be added to a suitable PAM stack;
`examples/pam-sudo` is a starting example. Keep password fallback available and test
`sudo` before a display manager. A PAM `sufficient` module can short-circuit other
`auth` modules, so review its placement if the stack has additional authentication
requirements. This project does not configure a second factor automatically.

Other commands:

```bash
sudo face-authd enroll --user "$USER" --device /dev/video2 --replace
sudo face-authd verify --user "$USER" --device /dev/video2 --frames 30
sudo face-authd enrolled
sudo face-authd unenroll --user "$USER"
face-authd probe-cameras
face-authd preview --device /dev/video2
```

## Migrating from dlib

**Re-enrollment is required.** dlib and SFace both emit 128 values, but their
features, normalization and thresholds are incompatible. The new model ID is
`yunet2023-sface2021-unit128-bgr-v1`; legacy encrypted/plaintext templates are
rejected with a re-enrollment message. Old files remain untouched until an explicit
`enroll --replace`/`setup`. A missing model ID is interpreted as legacy dlib, never
as SFace. New authenticated envelopes bind username and model ID through AES-GCM
associated data and use a new format version.

Do not reuse old dlib threshold settings. Review any `FACE_AUTHD_THRESHOLD` drop-in
and explicitly validate the chosen SFace threshold on your sensor. Re-enroll when
changing cameras or spectral modality, and test separate sessions, lighting,
head poses and glasses. A successful same-person test alone does not calibrate
the other-person false-accept rate.

## Configuration and hardware

Daemon environment settings can be configured with `sudo systemctl edit face-authd`:

```ini
[Service]
Environment=FACE_AUTHD_VIDEO_DEVICE=/dev/v4l/by-path/YOUR-CAMERA-video-index0
Environment=FACE_AUTHD_THRESHOLD=1.128
```

An explicit stable device path takes precedence and fails if unavailable. Otherwise
streaming GREY/Y8 capture nodes are preferred, followed by another streaming capture
node. Set the same camera for enrollment, verification, and daemon. Restart the
service after changing a drop-in. `verify --threshold` changes only that diagnostic
run, not the daemon. `verify` exits with an error if it does not observe two
consecutive matches. Thresholds must be finite and in `(0, 1.128]`.

Optional IR emitter helper settings:

```ini
[Service]
Environment=FACE_AUTHD_EMITTER_MODE=external-command
Environment=FACE_AUTHD_EMITTER_CMD=/absolute/path/to/helper
Environment=FACE_AUTHD_EMITTER_ARGS=run
```

Use the same helper settings while enrolling/testing. The helper must finish within
two seconds. Arguments are split on whitespace (there is no shell expansion),
and stdout/stderr are discarded. The current interface has no hardware-specific
emitter shutdown operation; verify the helper's lifecycle on your hardware. A
camera enumerating a GREY stream does not confirm illumination is working.

Defaults:

- Data and key: `/var/lib/face-authd` and `enrollment.key`.
- Models: `/var/lib/face-authd/models` (`FACE_AUTHD_MODEL_DIR` overrides it).
- Test data directory: `FACE_AUTHD_DATA_DIR` may select a separate store.
- Socket: `/run/face-authd.sock` (`FACE_AUTHD_SOCKET` overrides it for the daemon;
  the PAM module always uses the compiled default).

## Authentication boundaries and storage

- The daemon runs as root. The socket is `0600`, in a root-owned non-writable
  directory. Both endpoints verify a root peer UID. Unprivileged PAM applications
  cannot use this interface.
- Allowed services are `sudo`, `sudo-i`, `gdm-password`, `login`, `sddm`, `kde`,
  `polkit-1`. Nonempty `rhost` is denied. These are assertions by a trusted root
  client, not independent proof of an active local session.
- One request per connection, newline terminated, maximum 8 KiB. Clients keep
  their write side open while waiting. Disconnects/extra input cancel the request.
  Read/write phases have two-second limits, active connections are capped at 32,
  and only one camera session runs at a time.
- Recognition has an eight-second deadline and cooperative cancellation. The
  capture phase now permits a four-second attempt window with five warmup frames
  instead of stopping after 15 frames, to accommodate camera restart variability. Native
  inference/blocked kernel calls cannot be forcibly interrupted; the camera permit
  remains held until the real worker exits. A busy device returns an error so PAM
  can proceed according to its configured fallback.
- Model failures stop startup; production has no stub provider or test bypass.
- Enrollment uses AES-256-GCM, owner-only directories, private atomic writes,
  serialized key creation, and model/account/descriptor validation. Back up keys
  with templates; losing the key prevents decryption. File encryption does not
  protect against someone who controls root.

## Packaging and checks

```bash
bash scripts/build-deb.sh   # native Debian/Ubuntu with sufficiently new OpenCV
bash scripts/build-rpm.sh   # native RPM distro, fpm and rpm required
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked
```

Packaging downloads/verifies the same pinned ONNX files and includes their license
texts. It resolves the host's PAM library path and installed OpenCV runtime packages;
relabeling native binaries as a different architecture is rejected. Ordinary Debian
removal retains enrollment and keys; `purge` deletes them. See `CONTRIBUTING.md`
for the native model smoke test and hardware release checks.

## License

Project code: MIT. Model license notices are listed above.
