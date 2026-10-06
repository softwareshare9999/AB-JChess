#!/usr/bin/env bash
# Download the V11 EvalFile shipped with official AB-JChess releases.
set -euo pipefail

DEST="${1:-nnue}"
NNUE_NAME="${NNUE_NAME:-abjchess-20261004.nnue}"
RELEASE="${NNUE_RELEASE:-v0.2c}"
ASSET="${NNUE_ASSET:-0.2c_bmi2.zip}"
REPO="${NNUE_REPO:-lxsgx23/AB-JChess}"

mkdir -p "$DEST"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
curl -fsSL -o "$workdir/abjchess-nnue.zip" \
  "https://github.com/${REPO}/releases/download/${RELEASE}/${ASSET}"
unzip -o "$workdir/abjchess-nnue.zip" -d "$workdir/unpack"
find "$workdir/unpack" -name "$NNUE_NAME" -exec cp -f {} "$DEST/" \;
test -f "$DEST/$NNUE_NAME"
ls -la "$DEST/$NNUE_NAME"
