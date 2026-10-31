#!/usr/bin/env bash
# Download the EvalFile shipped with the upstream AB-JChess release that matches
# this source tree. The net name comes from ABJ_NET_FILE in abjchess/src/uci.cpp,
# so it follows upstream automatically after a merge. Upstream publishes a single
# zip per release (containing every platform binary plus the net), so the
# release zip is located through the GitHub API rather than by guessing a name.
#
# Env overrides: NNUE_NAME, NNUE_REPO, NNUE_TAG, NNUE_ASSET.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/abjchess/src"

src_name="$(sed -n 's/.*#define *ABJ_NET_FILE *"\([^"]*\)".*/\1/p' "$SRC/uci.cpp" | head -n1)"

DEST="${1:-nnue}"
NNUE_NAME="${NNUE_NAME:-$src_name}"
REPO="${NNUE_REPO:-lxsgx23/AB-JChess}"

if [ -z "$NNUE_NAME" ]; then
  echo "Could not determine the net name (ABJ_NET_FILE) from $SRC/uci.cpp" >&2
  exit 1
fi

echo "Looking for $NNUE_NAME in $REPO releases"

mkdir -p "$DEST"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

if [ -n "${NNUE_ASSET:-}" ]; then
  # Explicit asset name, e.g. AB-JChess-20261031.zip
  url="https://github.com/${REPO}/releases/download/${NNUE_TAG:?NNUE_TAG is required with NNUE_ASSET}/${NNUE_ASSET}"
else
  api="https://api.github.com/repos/${REPO}/releases/"
  api="${api}${NNUE_TAG:+tags/$NNUE_TAG}"
  api="${api}${NNUE_TAG:-latest}"
  # First browser_download_url ending in .zip is the release package.
  url="$(curl -fsSL "$api" \
    | tr ',' '\n' \
    | sed -n 's/.*"browser_download_url": *"\([^"]*\.zip\)".*/\1/p' \
    | head -n1)"
fi

if [ -z "$url" ]; then
  echo "No release zip found for $REPO (set NNUE_TAG / NNUE_ASSET to pin one)" >&2
  exit 1
fi

echo "Fetching $NNUE_NAME from $url"
curl -fsSL -o "$workdir/abjchess-release.zip" "$url"
if command -v unzip >/dev/null 2>&1; then
  unzip -oq "$workdir/abjchess-release.zip" -d "$workdir/unpack"
else
  # Minimal runners may lack unzip; Python 3 is always present on the CI images.
  python3 -m zipfile -e "$workdir/abjchess-release.zip" "$workdir/unpack"
fi
find "$workdir/unpack" -name "$NNUE_NAME" -exec cp -f {} "$DEST/" \;
test -f "$DEST/$NNUE_NAME"
ls -la "$DEST/$NNUE_NAME"

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "NNUE_NAME=$NNUE_NAME" >> "$GITHUB_ENV"
fi
