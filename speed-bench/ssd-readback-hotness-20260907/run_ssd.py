#!/usr/bin/env python3
"""Serial resident / actual bounded expert-cache experiment. Download paused only here."""
import argparse, csv, hashlib, json, os, signal, subprocess
from pathlib import Path

work=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
main=Path('/Users/brianfarley/Desktop/ds4')
prompt=Path('/Users/brianfarley/Desktop/ds4-qwen-iq2-wide/speed-bench/qwen-iq2-wide-20260907/fixed-corpus.txt')
models={'glm':main/'gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf', 'ds4':main/'ds4flash.gguf'}
parser=argparse.ArgumentParser()
parser.add_argument('--timing',action='store_true',help='omit full-logit disk writes from timed generation')
args=parser.parse_args()
pid=83728
paused=False
results=[]
try:
 command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout
 if 'curl -fL' in command and 'Qwen3.8-Flash-Next-IQ2XXSImatrix' in command:
  os.kill(pid,signal.SIGSTOP); paused=True
 for model,path in models.items():
  # Same sequence and binaries; SSD cache size is explicit, no simulated RAM claim.
  for mode,cache in [('base',None),('base',8),('candidate',8),('candidate',32)]:
   tag=f'{model}-{mode}-{cache or "resident"}'+('-timing' if args.timing else '')
   root=main if mode=='base' else work
   logits=out/f'{tag}-logits'; logits.mkdir(exist_ok=True)
   cmd=[str(root/'ds4-bench'),'-m',str(path),'--prompt-file',str(prompt),
        '--ctx-start','2048','--ctx-max','2048','--ctx-alloc','4096','--gen-tokens','64',
        '--teacher-forced-decode','--csv',str(out/f'{tag}.csv')]
   if not args.timing: cmd+=['--dump-decode-logits-dir',str(logits)]
   if cache: cmd+=['--ssd-streaming','--ssd-streaming-cache-experts',f'{cache}GB']
   print('START',tag,flush=True)
   with (out/f'{tag}.log').open('w') as log:
    run=subprocess.run(cmd,cwd=root,stdout=log,stderr=log,timeout=900)
   row={'tag':tag,'exit':run.returncode}
   if run.returncode==0:
    row['csv']=list(csv.DictReader((out/f'{tag}.csv').open()))
    row['logits']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in logits.iterdir() if p.is_file()}
   results.append(row)
   (out/('timing-results.json' if args.timing else 'results.json')).write_text(json.dumps(results,indent=2))
   print('DONE',tag,run.returncode,row.get('csv'),flush=True)
finally:
 if paused:
  os.kill(pid,signal.SIGCONT)
  print('Download resumed',flush=True)
