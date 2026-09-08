#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=${TMPDIR:-/tmp}/logan-ane-abi-probe.$$
trap 'rm -f "$OUT"' EXIT INT TERM HUP

xcrun clang -O2 "$HERE/abi_probe.m" -framework Foundation -ldl -o "$OUT"
exec "$OUT"
