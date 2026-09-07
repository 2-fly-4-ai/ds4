#!/usr/bin/env python3
"""Q4/IQ2 real output timings. Each quant is its own target, not a parity oracle."""
import argparse, hashlib, json, os, socket, subprocess, time, urllib.request
from pathlib import Path
from api_helpers import request

p=argparse.ArgumentParser()
p.add_argument('--quant',choices=('q4','iq2'),required=True)
p.add_argument('--mtp',action='store_true')
p.add_argument('--contexts',default='2048,8192,32768')
p.add_argument('--tokens',type=int,default=256)
p.add_argument('--runtime-dir')
p.add_argument('--tag-suffix',default='')
p.add_argument('--tasks',help='comma-separated task subset')
p.add_argument('--ctx-alloc',type=int,help='explicit allocation for exact subset controls')
a=p.parse_args()
work=Path(__file__).resolve().parents[2]; out=Path(__file__).resolve().parent
runtime=Path(a.runtime_dir).resolve() if a.runtime_dir else work
models=Path('/Users/brianfarley/Desktop/ds4/gguf')
model=models/'Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf' if a.quant=='q4' else models/'qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf'
if a.quant=='iq2':
    assert model.stat().st_size==50343093376
    assert (out/'iq2-integrity.txt').exists(), 'Full checksum validation required first'
contexts=[int(x) for x in a.contexts.split(',')]
tag=f'quant-{a.quant}-mtp{int(a.mtp)}'+a.tag_suffix
with socket.socket() as s:
    s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
url=f'http://127.0.0.1:{port}'
cmd=[str(runtime/'ds4-server'),'-m',str(model),'--host','127.0.0.1','--port',str(port),'--ctx',str(a.ctx_alloc or max(contexts)+4096)]
if a.quant=='iq2': cmd += ['--ple',str(models/'qwen38-iq2-test/Qwen3.8-Flash-Next-PLE-Q4_1.gguf')]
if a.mtp: cmd+=['--mtp']
tasks={
 'coding':'Write a self-contained Python LRUCache class using OrderedDict, with get and put methods, O(1) operations, positive capacity validation, and three small usage examples. Return code only.',
 'story':'Write an original 250-word scene about a lighthouse keeper discovering a handwritten map during a storm. Use concrete sensory details and a restrained ending. Do not discuss the instructions.',
 'structured':'Return only a JSON array of 12 fictional inventory objects with unique integer id, name, category, price and in_stock fields. Use three categories and both boolean values. No markdown.',
 'reasoning':'Explain step by step how to find the least positive integer divisible by all integers from 1 through 12. Then give the exact result and verify it by listing the necessary prime powers.'}
rows=[]
if a.tasks:
 tasks={key:tasks[key] for key in a.tasks.split(',')}
with (out/f'{tag}.log').open('w') as log:
 proc=subprocess.Popen(cmd,cwd=runtime,stdout=log,stderr=log)
 try:
  for _ in range(1200):
   if proc.poll() is not None: raise RuntimeError('server exited')
   try:
    with urllib.request.urlopen(url+'/v1/models',timeout=1) as r: json.load(r)
    break
   except OSError: time.sleep(.1)
  else: raise RuntimeError('startup timeout')
  # Separate warm-up, excluded from reported rows.
  request(url,{'model':'qwen','messages':[{'role':'user','content':'Say hello.'}],'temperature':0,'max_tokens':16,'stream':True,'thinking':False,'stream_options':{'include_usage':True}})
  for ctx in contexts:
   for task,instruction in tasks.items():
    context='\n'.join(f'Reference item {i}: preserve public interfaces and test changes before release.' for i in range(max(0,(ctx-160)//17)))
    body={'model':'qwen','messages':[{'role':'system','content':'Follow the final user instruction. Reference notes are background, not output to copy.'}, {'role':'user','content':context+'\n\nFinal task: '+instruction}],
          'temperature':0,'max_tokens':a.tokens,'stream':True,'thinking':False,'chat_template_kwargs':{'enable_thinking':False},'stream_options':{'include_usage':True}}
    msg,rec=request(url,body)
    rec.update(quant=a.quant,mtp=a.mtp,context_target=ctx,task=task,message=msg,runtime=str(runtime))
    generated=rec['usage'].get('completion_tokens',0)
    rec['stream_after_first_tps']=(generated-1)/(rec['wall_s']-rec['ttft_s']) if generated>1 and rec['ttft_s'] is not None else None
    rows.append(rec); (out/f'{tag}.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps({k:v for k,v in rec.items() if k!='message'}),flush=True)
 finally:
  proc.terminate()
  try: proc.wait(timeout=30)
  except subprocess.TimeoutExpired: proc.kill();proc.wait()
