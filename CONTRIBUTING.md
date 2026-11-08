# Contributing

Thanks for your interest in improving `face-authd`.

## Development Setup

OpenCV >=4.8 or OpenCV 5 development files are required; see README for native
libraries and the pinned OpenCV 4.10 CI build. No dlib dependency is used.


```bash
git clone <your-fork-or-repo-url>
cd face-authd
cargo check -p pam-face-auth -p face-authd
```

For full release builds:

```bash
cargo build --release -p pam-face-auth -p face-authd
```

## Before Opening a PR

- Keep changes focused and scoped.
- Run at least:
  - `cargo fmt --all -- --check`
  - `cargo test --workspace --locked`
  - `cargo clippy --workspace --all-targets --locked`
- If you changed packaging:
  - `./scripts/build-deb.sh`
- If you changed auth behavior:
  - test `face-authd setup` and `face-authd verify` on real hardware.

## Commit / PR Guidelines

- Use clear commit messages with intent.
- In PR description include:
  - what changed
  - why it changed
  - how it was tested
  - any breaking behavior

## Code Style

- Follow existing Rust style in the repository.
- Prefer small, explicit changes over broad refactors.
- Keep security-sensitive code paths easy to review.

## Areas Where Help Is Welcome

- camera compatibility across more devices
- reliability of PAM integration on more distros
- key management hardening
- packaging and upgrade safety
- tests and CI coverage

## Native model smoke test

The regular suite covers scoring, consecutive decisions, protocol policy and
framing/timeouts, encrypted storage and key races, and model integrity. It does
not need cameras or production enrollment. To also test real native model loading:

```bash
FACE_AUTHD_MODEL_DIR="$PWD/target/test-models" cargo run --locked -p face-authd -- fetch-models
FACE_AUTHD_MODEL_DIR="$PWD/target/test-models" cargo test --locked -p face-authd real_models_load_and_blank_frame_has_no_face -- --ignored
```

The native smoke test runs YuNet on a synthetic blank frame and exercises SFace alignment, inference and unit normalization on synthetic landmarks. It does not measure biometric
accuracy or exercise a real PAM stack; hardware acceptance/rejection and password
fallback remain separate release checks.
