#!/bin/sh
# Pass an unchanged pre-port misc shader from base 3a9ebe6; do not run after
# replacing that file. The host binary and every other shader stay identical.
set -eu
old=${1:?pass the pre-port dsv4_misc.metal path}
if [ "$(git hash-object "$old")" != "$(git rev-parse 3a9ebe6:metal/dsv4_misc.metal)" ]; then
 echo "The old shader must match base 3a9ebe6 exactly." >&2
 exit 2
fi
main=/Users/brianfarley/Desktop/ds4
glm=$main/gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf
out=speed-bench/attention-prefill-results
shasum -a 256 "$old" metal/dsv4_misc.metal > "$out/barrier-source-hashes.txt"
for n in 2048 8192; do
 run=0
 for label in old new new old; do
  run=$((run+1))
  if [ "$label" = old ]; then export DS4_METAL_DSV4_MISC_SOURCE="$old"; else unset DS4_METAL_DSV4_MISC_SOURCE; fi
  ./speed-bench/fixed_teacher_bench "$glm" "$main/ds4.c" "$n" "$label" > "$out/glm-barrier-$n-$run-$label.log" 2>&1
  echo "barrier cost $n $run $label exit=0"
 done
done
