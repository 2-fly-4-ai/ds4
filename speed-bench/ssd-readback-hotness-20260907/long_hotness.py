#!/usr/bin/env python3
"""Longer ABBA SSD cache-adaptation timing, without timed logit writes."""
import csv, json, os, subprocess
from pathlib import Path
work=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
main=Path('/Users/brianfarley/Desktop/ds4')
prompt=Path('/Users/brianfarley/Desktop/ds4-qwen-iq2-wide/speed-bench/qwen-iq2-wide-20260907/fixed-corpus.txt')
env=dict(os.environ,DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY='1')
results=[]
for index,mode in enumerate(('base','candidate','candidate','base')):
 tag=f'ds4-hotness-{index}-{mode}'
 root=main if mode=='base' else work
 cmd=[str(root/'ds4-bench'),'-m',str(main/'ds4flash.gguf'),'--prompt-file',str(prompt),
      '--ctx-start','2048','--ctx-max','2048','--ctx-alloc','4096','--gen-tokens','512',
      '--teacher-forced-decode','--ssd-streaming','--ssd-streaming-cache-experts','8GB',
      '--csv',str(out/f'{tag}.csv')]
 print('START',tag,flush=True)
 with (out/f'{tag}.log').open('w') as log:
  run=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=log,timeout=900)
 row={'tag':tag,'exit':run.returncode}
 if run.returncode==0:row['csv']=list(csv.DictReader((out/f'{tag}.csv').open()))
 results.append(row);(out/'long-hotness-results.json').write_text(json.dumps(results,indent=2))
 print('DONE',row,flush=True)
