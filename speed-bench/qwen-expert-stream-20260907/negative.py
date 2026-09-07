#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
out=Path(__file__).resolve().parent; work=out.parents[1]
main=Path('/Users/brianfarley/Desktop/ds4')
iq=main/'gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf'
ple=iq.parent/'Qwen3.8-Flash-Next-PLE-Q4_1.gguf'
cases=[('q4',main/'gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf',[],
        'requires IQ2_XXS'),
       ('small-cache',iq,['--ssd-streaming-cache-experts','79'],'at least 80 experts'),
       ('cold',iq,['--ssd-streaming-cold'],'does not yet support cold')]
results=[]
for tag,model,extra,expected in cases:
    cmd=[str(work/'ds4'),'-m',str(model),'--ssd-streaming','--ssd-streaming-cache-experts','8GB',
         '--ctx','1024','-n','1','-p','test']
    if model==iq: cmd+=['--ple',str(ple)]
    cmd+=extra
    p=subprocess.run(cmd,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    (out/('negative-'+tag+'.log')).write_text(p.stdout)
    ok=p.returncode!=0 and expected in p.stdout
    results.append(dict(tag=tag,exit=p.returncode,passed=ok,command=cmd))
    print(tag,ok,flush=True)
    assert ok,p.stdout
(out/'negative-results.json').write_text(json.dumps(results,indent=2))
