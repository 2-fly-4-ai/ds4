"""Serial, unchanged-model CLI regression checks against the production build."""
import hashlib, os, subprocess
from pathlib import Path

build=Path(__file__).resolve().parents[1]
common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],cwd=build,text=True).strip()).parent
root=Path(os.environ.get('QWEN_PORT_MODEL_ROOT',str(common)))
baseline=Path(os.environ.get('QWEN_PORT_BASELINE',str(common)))
if baseline.resolve()==build.resolve():raise SystemExit('Set QWEN_PORT_BASELINE to a separate baseline build, not the candidate itself')
out=build/'speed-bench/qwen-mtp-port-20260907/cross-model'
out.mkdir(parents=True,exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4_','HOT_'))}
for family,model,extra in (
    ('ds4',root/'ds4flash.gguf',[]),
    ('glm',root/'gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf',[]),
    ('ds4-ssd',root/'ds4flash.gguf',['--ssd-streaming','--ssd-streaming-cache-experts','8GB']),
):
    outputs=[]
    for name,cwd in (('base',baseline),('candidate',build)):
        print('START',family,name,flush=True)
        with (out/f'{family}-{name}.log').open('xb') as log:
            p=subprocess.run([str(cwd/'ds4'),'-m',str(model),'--metal','--ctx','2048','-n','32',
                '--temp','0','--nothink','-p','Write a Python function to add two integers. Briefly explain it.',*extra],
                cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=log,timeout=600,check=True)
        with (out/f'{family}-{name}.out').open('xb') as f:f.write(p.stdout)
        outputs.append(p.stdout)
    if outputs[0]!=outputs[1]:raise RuntimeError(f'{family}: output mismatch')
    print(f'{family}: PASS {hashlib.sha256(outputs[0]).hexdigest()}',flush=True)
