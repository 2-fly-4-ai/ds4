#!/usr/bin/env python3
"""ABBA preservation checks, serialized across all models."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
out=Path(__file__).resolve().parent; work=out.parents[1]
main=Path('/Users/brianfarley/Desktop/ds4')
arms=[('qwen-resident','gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf',False),
      ('glm-resident','gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf',False),
      ('ds4-resident','ds4flash.gguf',False),
      ('glm-stream','gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf',True),
      ('ds4-stream','ds4flash.gguf',True)]
results=[]
reverse = '--ds4-reverse' in sys.argv
steady = '--ds4-steady' in sys.argv
parity = '--parity' in sys.argv
if parity: arms=[a for a in arms if a[2]]
if reverse or steady: arms=[a for a in arms if a[0]=='ds4-resident']
prefix = 'parity-' if parity else 'reverse-' if reverse else 'steady-' if steady else ''
for name,model,stream in arms:
    hashes=[]
    roots=[main,work] if parity else [work,main,main,work] if reverse else [main,work,work,main]
    for i,root in enumerate(roots):
        tag=f'guard-{prefix}{name}-{i}'
        cmd=[str(root/'ds4-bench'),'-m',str(main/model),'--prompt-file',
             str(main/'speed-bench/qwen-iq2-wide-20260907/fixed-corpus.txt'),
             '--ctx-start','2048','--ctx-max','2048','--ctx-alloc','4096',
             '--gen-tokens','64' if parity else '512' if steady else '128','--teacher-forced-decode','--csv',str(out/(tag+'.csv'))]
        if stream: cmd+=['--ssd-streaming','--ssd-streaming-cache-experts','8GB']
        if parity:
            dump=out/(tag+'-logits'); dump.mkdir(exist_ok=True)
            cmd+=['--dump-decode-logits-dir',str(dump)]
        print('START',tag,flush=True)
        with (out/(tag+'.log')).open('w') as log:
            p=subprocess.run(cmd,cwd=root,stdout=log,stderr=log,timeout=900)
        row=dict(tag=tag,command=cmd,exit=p.returncode)
        if p.returncode==0: row['csv']=list(csv.DictReader((out/(tag+'.csv')).open()))
        if p.returncode==0 and parity:
            row['hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(dump.glob('*.f32'))}
            assert len(row['hashes'])==64
            hashes.append(row['hashes'])
        results.append(row); (out/('guard-'+prefix+'results.json')).write_text(json.dumps(results,indent=2))
        print(row,flush=True)
        if p.returncode: raise SystemExit(1)
    if parity: assert hashes[0]==hashes[1],name+' full-logit mismatch'
