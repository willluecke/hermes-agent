#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-/home/will/src/hermes-agent/ops/command-center}"
BIN_DIR="${BIN_DIR:-/home/will/.local/bin}"
LIBEXEC_DIR="${LIBEXEC_DIR:-/home/will/.local/libexec}"

install -d -m 0755 "$BIN_DIR" "$LIBEXEC_DIR"
install -m 0755 \
  "$SOURCE_DIR/restart-gateway-when-idle.sh" \
  "$BIN_DIR/hermes-command-center-restart-gateway"
install -m 0755 \
  "$SOURCE_DIR/restart-gateway-broker-worker.sh" \
  "$LIBEXEC_DIR/hermes-command-center-restart-worker"

printf 'Installed command-center gateway restart broker launcher and worker.\n'
