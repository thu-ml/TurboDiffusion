#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python "$ROOT/turbodiffusion/inference/h3_i2va_infer.py" "$@"
