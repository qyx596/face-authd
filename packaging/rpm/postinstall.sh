#!/usr/bin/env bash
set -euo pipefail

if command -v systemctl >/dev/null 2>&1; then
  systemctl daemon-reload || true
  if [ -d /run/systemd/system ]; then
    systemctl enable --now face-authd.service || true
  fi
fi
