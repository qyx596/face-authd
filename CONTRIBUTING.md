# Contributing

Thanks for your interest in improving `face-authd`.

## Development Setup

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
  - `cargo check -p pam-face-auth -p face-authd`
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
