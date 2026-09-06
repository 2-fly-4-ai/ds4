"""Sequential paired full-logit/state checks; no simultaneous model jobs."""
import json, os, subprocess, sys
from pathlib import Path
build=Path(__file__).resolve().parents[1]
common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],cwd=build,text=True).strip()).parent
root=Path(os.environ.get('QWEN_PORT_MODEL_ROOT',str(common)))
out=build/'speed-bench/qwen-mtp-port-20260907';out.mkdir(parents=True,exist_ok=True)
if '--label' in sys.argv:
    out=out/sys.argv[sys.argv.index('--label')+1];out.mkdir(parents=True,exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','HOT_'))}
tasks={
 'code':'Write a Python function that detects a cycle in a directed graph and returns one actual cycle. Handle disconnected graphs. Include tests and explain complexity.',
 'story':'Write a vivid original story about a retired cartographer who discovers a city that moves every night. Use dialogue and an unexpected but coherent ending.',
 'json':'Return only a JSON array of 12 distinct test cases for a calendar booking API. Each object needs an id, scenario, input, expected_status and reason. Include concurrency, timezone, and validation cases.',
 'copy':(root/'speed-bench/qwen_prompt_lookup_copy_chat.txt').read_text(),
 'edit':(root/'speed-bench/qwen_prompt_lookup_edit_chat.txt').read_text()}
jobs=[(v,'code',2048,0,False) for v in [8]]
if '--variants' in sys.argv:jobs=[(int(v),'code',2048,0,False) for v in sys.argv[sys.argv.index('--variants')+1].split(',')]
if '--wide' in sys.argv:
    variants=[int(v) for v in sys.argv[sys.argv.index('--wide')+1].split(',')]
    jobs=[(v,t,c,temp,lookup) for v in variants for t,c,temp,lookup in
          [('story',2048,0,False),('json',2048,0.7,False),('copy',2048,0,True),
           ('edit',2048,0,True),('code',8192,0,False),('edit',8192,0,True),('code',32768,0,False)]]
failed=[]
for v,task,ctx,temp,lookup in jobs:
    name=f'v{v}-{task}-{ctx}-t{temp}-lookup{int(lookup)}'
    e=dict(env,HOT_TASK=tasks[task],HOT_CTX=str(ctx),HOT_VARIANT=str(v),HOT_TEMP=str(temp))
    if ctx>=8192:e['HOT_PAD']='1'
    if lookup:e['HOT_LOOKUP']='1'
    if '--timing' in sys.argv:e['HOT_TIMING']='1'
    if '--steady' in sys.argv:e['HOT_STEADY']='1'
    if '--plain' in sys.argv:e['HOT_PLAIN']='1'
    print('START',name,flush=True)
    with (out/(name+'.log')).open('xb') as f:
        p=subprocess.run([str(build/'tests/test_qwen_mtp_port'),str(root/'gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf')],cwd=build,env=e,stdout=f,stderr=f,timeout=600)
    print('PASS' if p.returncode==0 else 'FAIL',name,flush=True)
    if p.returncode:failed.append(name)
print('FAILURES',json.dumps(failed),flush=True)
sys.exit(bool(failed))
