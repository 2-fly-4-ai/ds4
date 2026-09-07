#!/usr/bin/env python3
import hashlib, json, subprocess
from pathlib import Path
out=Path(__file__).resolve().parent
model=Path('/Users/brianfarley/Desktop/ds4/gguf/qwen38-iq2-test')
prompts={
 'code':'Write a Python LRU cache class with get and put methods, type hints, and unit tests.\n',
 'story':'Write a story about a lighthouse keeper who receives a letter from tomorrow.\n',
 'json':'Produce a JSON array of twelve fictional books, each with title, author, year, and genre.\n',
}
results=[]
for task,prompt in prompts.items():
    hashes=[]
    for variant,binary,budget in [('base','/tmp/ds4-qwen-base-gen',0),('stream','/tmp/ds4-qwen-stream-gen',8)]:
        tag=f'mtp-{task}-{variant}'
        trace=out/(tag+'.bin')
        cmd=[binary,str(model/'Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf'),
             str(model/'Qwen3.8-Flash-Next-PLE-Q4_1.gguf'),str(budget),'1','48',prompt,str(trace)]
        with (out/(tag+'.log')).open('w') as log:
            p=subprocess.run(cmd,stdout=log,stderr=log,timeout=600)
        row=dict(tag=tag,exit=p.returncode,command=cmd)
        if p.returncode==0:
            row['sha256']=hashlib.sha256(trace.read_bytes()).hexdigest()
            hashes.append(row['sha256'])
        results.append(row)
        (out/'generation-results.json').write_text(json.dumps(results,indent=2))
        print(tag,p.returncode,flush=True)
        if p.returncode: raise SystemExit(1)
    print(task,'full trace identical:',hashes[0]==hashes[1],flush=True)
    if hashes[0]!=hashes[1]: raise SystemExit(2)
