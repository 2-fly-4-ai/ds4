#!/usr/bin/env python3
"""Cold-context teacher-forced quant comparison; timing excludes logit dumping."""
import argparse,csv,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--quant',choices=('q4','iq2'),required=True)
p.add_argument('--rounds',type=int,default=2);a=p.parse_args()
work=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
models=Path('/Users/brianfarley/Desktop/ds4/gguf')
model=models/'Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf';extra=[]
if a.quant=='iq2':
 assert (out/'iq2-integrity.txt').exists()
 model=models/'qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf'
 extra=['--ple',str(models/'qwen38-iq2-test/Qwen3.8-Flash-Next-PLE-Q4_1.gguf')]
rows=[]
for rep in range(a.rounds):
 for ctx in (2048,8192,32768):
  tag=f'fixed-{a.quant}-{rep}-{ctx}'
  cmd=[str(work/'ds4-bench'),'-m',str(model),*extra,'--prompt-file',str(out/'fixed-corpus.txt'),
       '--ctx-start',str(ctx),'--ctx-max',str(ctx),'--ctx-alloc',str(ctx+4096),
       '--gen-tokens','128','--teacher-forced-decode','--csv',str(out/f'{tag}.csv')]
  print('START',tag,flush=True)
  with (out/f'{tag}.log').open('w') as log:
   run=subprocess.run(cmd,cwd=work,stdout=log,stderr=log,timeout=600)
  assert run.returncode==0,(tag,run.returncode)
  row={'quant':a.quant,'rep':rep,'ctx':ctx,'data':list(csv.DictReader((out/f'{tag}.csv').open()))}
  rows.append(row);(out/f'fixed-{a.quant}.json').write_text(json.dumps(rows,indent=2))
  print(row,flush=True)
