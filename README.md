# face-authd
[![Release](https://img.shields.io/github/v/release/qyx596/face-authd?display_name=tag)](https://github.com/qyx596/face-authd/releases)
[![License](https://img.shields.io/github/license/qyx596/face-authd)](https://github.com/qyx596/face-authd/blob/main/LICENSE)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Linux-blue.svg)](https://www.kernel.org/)

Rust-based local face authentication for Linux via PAM.

`face-authd` consists of:

- `face-authd` daemon (camera + recognition)
- `pam_face_auth.so` PAM module
- Debian packaging scripts
- RPM packaging script

## What It Uses

This project currently uses **dlib** face models:

- `shape_predictor_5_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

The `.deb` package bundles these models into:

- `/var/lib/face-authd/models`

## Quick Start (Recommended)

### 1) Download package from GitHub Releases

Download the latest package for your distro:

- Debian/Ubuntu: `face-authd_<version>_amd64.deb`
- Fedora/RHEL/openSUSE: `face-authd-<version>-1.<arch>.rpm`

### 2) Install

Debian/Ubuntu:

```bash
sudo dpkg -i ./face-authd_<version>_amd64.deb
```

Fedora/RHEL/openSUSE:

```bash
sudo rpm -Uvh ./face-authd-<version>-1.<arch>.rpm
```

For `.deb`, debconf can ask whether to auto-configure PAM.

### 3) Run setup (required)

Enroll + verify in one command:

```bash
sudo face-authd setup
```

Optional:

```bash
sudo face-authd setup --user "$USER" --device /dev/video2
```

## Manual Install (Any Linux Distro)

If you are not using Debian-based systems, install manually:

```bash
cargo build --release -p pam-face-auth -p face-authd
```

Install daemon and service:

```bash
sudo install -Dm755 target/release/face-authd /usr/local/bin/face-authd
sudo install -Dm644 systemd/face-authd.service /etc/systemd/system/face-authd.service
sudo systemctl daemon-reload
sudo systemctl enable --now face-authd
```

Install PAM module (`pam_face_auth.so`):

- Debian/Ubuntu:
  - `/lib/x86_64-linux-gnu/security/pam_face_auth.so`
- Fedora/RHEL/CentOS/openSUSE (typical):
  - `/usr/lib64/security/pam_face_auth.so`

Example (Fedora/RHEL style):

```bash
sudo install -Dm755 target/release/libpam_face_auth.so /usr/lib64/security/pam_face_auth.so
```

Then add `pam_face_auth.so` to your PAM stack (`/etc/pam.d/sudo` first, then login stack if needed).

## Most Useful Commands

```bash
# Enroll only
sudo face-authd enroll --user "$USER" --replace

# Verify only
sudo face-authd verify --user "$USER" --device /dev/video2

# Enroll + verify (advanced version of setup)
sudo face-authd enroll-verify --user "$USER" --device /dev/video2 --replace

# List enrolled users
face-authd enrolled
```

## Storage

- Data: `/var/lib/face-authd`
- Models: `/var/lib/face-authd/models`

Enrollment templates are encrypted at rest (AES-256-GCM). Encryption keys are stored in Linux keyring.

## Notes

- Keep password fallback enabled while testing PAM.
- Test with `sudo` first, then GDM login.
- Project is experimental.

## License

MIT
