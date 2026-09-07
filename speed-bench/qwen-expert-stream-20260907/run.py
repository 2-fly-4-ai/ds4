#!/usr/bin/env python3
"""Serial experiments; original main is the unmodified resident reference."""
import csv, hashlib, json, subprocess, sys
from pathlib import Path

out = Path(__file__).resolve().parent
work = out.parents[1]
main = Path('/Users/brianfarley/Desktop/ds4')
model = main/'gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf'
ple = model.parent/'Qwen3.8-Flash-Next-PLE-Q4_1.gguf'
tag, build, context, count, cache = sys.argv[1:]
root = main if build == 'base' else work
dump = not tag.startswith('timing')
cmd = [str(root/'ds4-bench'), '-m', str(model), '--ple', str(ple),
       '--prompt-file', str(main/'speed-bench/qwen-iq2-wide-20260907/fixed-corpus.txt'),
       '--ctx-start', context, '--ctx-max', context, '--ctx-alloc', str(int(context)+1024),
       '--gen-tokens', count, '--teacher-forced-decode', '--csv', str(out/f'{tag}.csv')]
if cache != '0': cmd += ['--ssd-streaming', '--ssd-streaming-cache-experts', cache+'GB']
if dump:
    dest = out/f'{tag}-logits'
    dest.mkdir(exist_ok=True)
    cmd += ['--dump-decode-logits-dir', str(dest)]
if 'memory' in tag:
    cmd = ['/usr/bin/time', '-l', *cmd]
with (out/f'{tag}.log').open('w') as log:
    p = subprocess.run(cmd, cwd=root, stdout=log, stderr=log, timeout=900)
row = dict(tag=tag, command=cmd, exit=p.returncode)
if p.returncode == 0:
    row['csv'] = list(csv.DictReader((out/f'{tag}.csv').open()))
    if dump:
        row['hashes'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in sorted(dest.glob('*.f32'))}
(out/f'{tag}.json').write_text(json.dumps(row, indent=2))
print(json.dumps(row), flush=True)
sys.exit(p.returncode)
