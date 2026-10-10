#!/usr/bin/env bash
# Download the V11 EvalFile shipped with the official AB-JChess release that
# matches this source tree. The net name comes from EvalFileDefaultNameBig in
# evaluate.h and the release tag from the UCI version in misc.cpp, so both
# follow upstream automatically after a merge.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/abjchess/src"

src_name="$(sed -n 's/.*EvalFileDefaultNameBig *= *"\([^"]*\)".*/\1/p' "$SRC/evaluate.h" | head -n1)"
src_version="$(sed -n 's/.*string_view version *= *"\([^"]*\)".*/\1/p' "$SRC/misc.cpp" | head -n1)"

DEST="${1:-nnue}"
NNUE_NAME="${NNUE_NAME:-$src_name}"
RELEASE="${NNUE_RELEASE:-v$src_version}"
ASSET="${NNUE_ASSET:-${src_version}_bmi2.zip}"
REPO="${NNUE_REPO:-lxsgx23/AB-JChess}"

if [ -z "$NNUE_NAME" ] || [ "$RELEASE" = "v" ]; then
  echo "Could not determine NNUE name or version from $SRC" >&2
  exit 1
fi

echo "Fetching $NNUE_NAME from $REPO release $RELEASE ($ASSET)"

mkdir -p "$DEST"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
curl -fsSL -o "$workdir/abjchess-nnue.zip" \
  "https://github.com/${REPO}/releases/download/${RELEASE}/${ASSET}"
unzip -o "$workdir/abjchess-nnue.zip" -d "$workdir/unpack"
find "$workdir/unpack" -name "$NNUE_NAME" -exec cp -f {} "$DEST/" \;
test -f "$DEST/$NNUE_NAME"
ls -la "$DEST/$NNUE_NAME"

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "NNUE_NAME=$NNUE_NAME" >> "$GITHUB_ENV"
fi
