#!/usr/bin/env python3
"""Bounded paired quality screen; not a full quality-equivalence claim."""
import argparse, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--quant',choices=('q4','iq2'),required=True)
a=p.parse_args()
work=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
models=Path('/Users/brianfarley/Desktop/ds4/gguf')
model=models/'Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf'
extra=[]
if a.quant=='iq2':
 assert (out/'iq2-integrity.txt').exists()
 model=models/'qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf'
 extra=['--ple',str(models/'qwen38-iq2-test/Qwen3.8-Flash-Next-PLE-Q4_1.gguf')]
cmd=[str(work/'ds4-eval'),'-m',str(model),*extra,'--questions','12',
     '--tokens','1536','--soft-limit-reply-budget','384','--hard-limit-reply-budget','192',
     '--temp','0','--seed','123','--ctx','8192','--plain','--pause-ms','1',
     '--trace',str(out/f'quality-{a.quant}.trace')]
print('First 12 embedded questions: GPQA / SuperGPQA / AIME; identical bounded thinking and answer budgets; no MTP.',flush=True)
with (out/f'quality-{a.quant}.log').open('w') as log:
 run=subprocess.run(cmd,cwd=work,stdout=log,stderr=log,timeout=1200)
print('eval exit',run.returncode,'(wrong benchmark answers are quality outcomes, not necessarily engine failures)',flush=True)
raise SystemExit(run.returncode)
