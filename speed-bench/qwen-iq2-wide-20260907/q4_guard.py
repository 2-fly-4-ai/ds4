#!/usr/bin/env python3
"""ABBA current-main/candidate throughput guard, same quant/context/tokens."""
import csv,json,subprocess
from pathlib import Path
work=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
main=Path('/Users/brianfarley/Desktop/ds4')
rows=[]
for rep,mode in enumerate(('main','candidate','candidate','main')):
 root=main if mode=='main' else work;tag=f'q4-guard-{rep}-{mode}'
 cmd=[str(root/'ds4-bench'),'-m',str(main/'gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf'),
      '--prompt-file',str(out/'fixed-corpus.txt'),'--ctx-start','2048','--ctx-max','2048',
      '--ctx-alloc','36864','--gen-tokens','512','--teacher-forced-decode','--csv',str(out/f'{tag}.csv')]
 print('START',tag,flush=True)
 with (out/f'{tag}.log').open('w') as log:
  run=subprocess.run(cmd,cwd=root,stdout=log,stderr=log,timeout=600)
 assert run.returncode==0,(tag,run.returncode)
 row={'mode':mode,'rep':rep,'data':list(csv.DictReader((out/f'{tag}.csv').open()))}
 rows.append(row);(out/'q4-guard.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
