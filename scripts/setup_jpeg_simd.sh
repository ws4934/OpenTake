#!/usr/bin/env bash
set -euo pipefail
# ARM uses the bundled NEON assembler; NASM is required only on x86.
case "$(uname -m)" in arm64|aarch64) exit 0 ;; esac
if command -v nasm >/dev/null; then exit 0; fi
case "$(uname -s)" in
  Linux)
    sudo apt-get update
    sudo apt-get install --yes --no-install-recommends nasm
    ;;
  Darwin) brew install nasm ;;
  *) echo 'Unsupported host for JPEG SIMD setup' >&2; exit 1 ;;
esac
