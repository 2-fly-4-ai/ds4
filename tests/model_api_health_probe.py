#!/usr/bin/env python3
"""Replay archived short requests through a candidate, preserving evidence."""
import json, os, socket, subprocess, sys, time, urllib.request, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],cwd=ROOT,text=True).strip()).parent
sys.path.insert(0, str(MAIN/'tests'))
from production_model_matrix import req, guard, memory
model = sys.argv[1]
phase = sys.argv[2] if len(sys.argv)>2 else 'qwen-chatml-completion'
source = MAIN/'speed-bench/production-matrix-20260914'/phase/model
out = ROOT/'speed-bench/api-health'/f'{model}-{os.environ.get("HEALTH_LABEL",phase)}'
out.mkdir(parents=True, exist_ok=True)
c=json.loads((source/'config.json').read_text()); args=c['argv'][:]
args[0]=str(Path(os.environ.get('HEALTH_BINARY_ROOT',str(ROOT)))/'ds4-server')
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
args[args.index('--port')+1]=str(port)
args[args.index('--trace')+1]=str(out/'trace.log')
env={k:v for k,v in os.environ.items() if not k.startswith('DS4_')}
env.update(c['env_overrides'])
if os.environ.get('HEALTH_MTP_OFF'):env['DS4_QWEN_NEXTN_DRAFT']='0'
(out/'config.json').write_text(json.dumps(dict(source_phase=phase,argv=args,env_overrides={k:v for k,v in env.items() if k.startswith('DS4_')}),indent=2))
with (out/'server.log').open('w') as log:
 p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=log)
 stop=threading.Event();fail=[]
 threading.Thread(target=guard,args=(p,stop,fail,memory()['swap_gib']),daemon=True).start()
 try:
  base=f'http://127.0.0.1:{port}';deadline=time.monotonic()+120
  while True:
   if p.poll() is not None:raise RuntimeError('server exited')
   try:
    with urllib.request.urlopen(base+'/v1/models',timeout=1) as r:inventory=json.load(r)
    break
   except OSError:
    if time.monotonic()>deadline:raise RuntimeError('startup timeout')
    time.sleep(.25)
  (out/'models.json').write_text(json.dumps(inventory,indent=2))
  if model.startswith('qwen27') and os.environ.get('HEALTH_ASSERT_ID'):
   assert inventory['data'][0]['id']=='qwen', inventory
  tasks=[(f'{context}-{task}',0) for context in os.environ.get('HEALTH_CONTEXTS','short').split(',')
         for task in os.environ.get('HEALTH_TASKS','code,story,reasoning,json,edit').split(',')]
  if os.environ.get('HEALTH_EXTRA'):
   tasks += [('short-json',0),('short-code',0.7),('short-story',0.7)]
  for index,(task,temperature) in enumerate(tasks):
   body=json.loads((source/f'{task}.request.json').read_text())
   if 'prompt' in body and os.environ.get('HEALTH_CHAT'):
    prompt=body.pop('prompt')
    prompt=prompt.removeprefix('<|im_start|>user\n').removesuffix('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    body['messages']=[dict(role='user',content=prompt)]
   body['temperature']=temperature
   if os.environ.get('HEALTH_THINK'):
    body['thinking']=True;body['chat_template_kwargs']={'enable_thinking':True}
   body['model']=inventory['data'][0]['id']
   result=req(base,body,120)
   result['temperature']=temperature;result['case']=task;result['thinking']=bool(os.environ.get('HEALTH_THINK'))
   (out/f'{index:02d}-{task}.json').write_text(json.dumps(result,indent=2))
   print(task,json.dumps(result),flush=True)
 finally:
  stop.set()
  p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
  if fail:raise RuntimeError(fail)
