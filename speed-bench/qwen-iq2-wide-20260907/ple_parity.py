#!/usr/bin/env python3
"""Exact same-weight embedded/external PLE comparison; no quant change."""
import hashlib, subprocess
from pathlib import Path
work=Path(__file__).resolve().parents[2]; out=Path(__file__).resolve().parent
model='/Users/brianfarley/Desktop/ds4/gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf'
ple='/Users/brianfarley/Desktop/ds4/gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-PLE-Q4_1.gguf'
for mode in ('embedded','external'):
 d=out/f'ple-{mode}-logits'; d.mkdir(exist_ok=True)
 cmd=[str(work/'ds4-bench'),'-m',model,'--prompt-file',str(out/'fixed-corpus.txt'),
      '--ctx-start','2048','--ctx-max','2048','--ctx-alloc','4096','--gen-tokens','128','--teacher-forced-decode',
      '--dump-decode-logits-dir',str(d),'--csv',str(out/f'ple-{mode}.csv')]
 if mode=='external': cmd += ['--ple',ple]
 with (out/f'ple-{mode}.log').open('w') as log:
  subprocess.run(cmd,cwd=work,stdout=log,stderr=log,check=True,timeout=600)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
left={p.name:digest(p) for p in (out/'ple-embedded-logits').iterdir() if p.is_file()}
right={p.name:digest(p) for p in (out/'ple-external-logits').iterdir() if p.is_file()}
assert len(left)>=128 and left==right,(len(left),len(right))
print(f'PASS: {len(left)} full-logit files byte-identical for embedded/external PLE')
