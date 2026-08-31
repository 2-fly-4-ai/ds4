#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
    echo "usage: $0 INPUT.gguf OUTPUT.gguf [--dry-run|--overwrite]" >&2
    exit 1
fi

INPUT=$1
OUTPUT=$2
shift 2

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TOOL="$ROOT/gguf-requantize-dense"
if [ ! -x "$TOOL" ]; then
    make -C "$ROOT" gguf-requantize-dense
fi

set -- --in "$INPUT" --out "$OUTPUT" --verify "$@"
LAYER=27
while [ "$LAYER" -le 42 ]; do
    set -- "$@" --tensor-type "blk.$LAYER.attn_output=q4_K"
    LAYER=$((LAYER + 1))
done

exec "$TOOL" "$@"
