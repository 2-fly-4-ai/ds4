#!/usr/bin/env python3
"""Local-only API smoke; always stop the owned server."""
import json,socket,subprocess,time,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent; work=out.parents[1]
model=Path('/Users/brianfarley/Desktop/ds4/gguf/qwen38-iq2-test')
base='http://127.0.0.1:18073'
with socket.socket() as probe:
    probe.bind(('127.0.0.1',18073))
cmd=[str(work/'ds4-server'),'-m',str(model/'Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf'),
     '--ple',str(model/'Qwen3.8-Flash-Next-PLE-Q4_1.gguf'),'--ctx','4096',
     '--ssd-streaming','--ssd-streaming-cache-experts','8GB','--mtp',
     '--host','127.0.0.1','--port','18073','--kv-disk-dir',str(out/'api-kv')]
with (out/'api-server.log').open('w') as log:
    p=subprocess.Popen(cmd,cwd=work,stdout=log,stderr=log)
    try:
        deadline=time.monotonic()+120
        while True:
            try:
                with urllib.request.urlopen(base+'/v1/models',timeout=2) as r: models=json.load(r)
                break
            except OSError:
                if p.poll() is not None or time.monotonic()>deadline: raise
                time.sleep(.25)
        messages=[{'role':'user','content':'Reply with only the word READY.'}]
        results=[]
        for i in range(2):
            body=dict(model=models['data'][0]['id'],messages=messages,temperature=0,
                      max_tokens=48,chat_template_kwargs={'enable_thinking':False})
            req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
                                       headers={'Content-Type':'application/json'})
            start=time.monotonic()
            with urllib.request.urlopen(req,timeout=120) as r: response=json.load(r)
            results.append(dict(seconds=time.monotonic()-start,response=response))
            choice=response['choices'][0]['message']
            assert choice.get('content'),response
            messages += [choice,{'role':'user','content':'Now reply with only DONE.'}]
        (out/'api-results.json').write_text(json.dumps(dict(command=cmd,results=results),indent=2))
        print('PASS: two API turns, SSD + MTP, nonempty responses',flush=True)
    finally:
        p.terminate()
        try: p.wait(timeout=20)
        except subprocess.TimeoutExpired: p.kill(); p.wait()
