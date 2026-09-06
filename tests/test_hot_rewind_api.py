"""Run only this server; exact-repeat regression across tasks and sampling."""
import hashlib, json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path
build, root, out = map(Path, sys.argv[1:]); out.mkdir(parents=True, exist_ok=True)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
url = f'http://127.0.0.1:{port}'
args = [str(build/'ds4-server'), '-m', str(root/'ds4flash.gguf'), '--host', '127.0.0.1', '--port', str(port), '--ctx', '8192']
if os.environ.get('HOT_SSD'):
    args += ['--ssd-streaming', '--ssd-streaming-cold', '--ssd-streaming-cache-experts', '4GB']
if os.environ.get('HOT_DSPARK'):
    args += ['--dspark', '--mtp-exact-sampling', '--mtp-model', str(root/'gguf/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf')]
with (out/'server.log').open('wb') as log:
    proc = subprocess.Popen(args, cwd=build, stdout=log, stderr=log)
    try:
        deadline = time.monotonic()+240
        while True:
            assert proc.poll() is None, 'server exited'
            try:
                with urllib.request.urlopen(url+'/v1/models', timeout=2) as r: model = json.load(r)['data'][0]['id']
                break
            except OSError:
                if time.monotonic()>deadline: raise RuntimeError('readiness timeout')
                time.sleep(.5)
        cases = [('code', 'Write a Python function to add two integers. Briefly explain it.'),
                 ('story', 'Write a short story about a lighthouse keeper who receives a letter from tomorrow.'),
                 ('json', 'Return only a JSON array of eight inventory items with id, name, quantity, and price.'),
                 ('copy', (root/'speed-bench/qwen_prompt_lookup_copy_chat.txt').read_text()),
                 ('edit', (root/'speed-bench/qwen_prompt_lookup_edit_chat.txt').read_text())]
        if os.environ.get('HOT_SSD'): cases = cases[:1]
        for label, prompt in cases:
            temps = [0] if os.environ.get('HOT_SSD') or os.environ.get('HOT_DSPARK') else [0, .7]
            for temp in temps:
                # Make the sampled variant a different prompt to exercise a
                # fresh prefill, followed by its exact cached repeat.
                content = prompt if temp == 0 else prompt + '\nPlease respond directly.'
                hashes = []
                for repeat in range(2):
                    body = dict(model=model, messages=[dict(role='user',content=content)],
                                temperature=temp, seed=1234, max_tokens=16 if os.environ.get('HOT_SSD') else 256,
                                stream=True, stream_options=dict(include_usage=True), chat_template_kwargs=dict(enable_thinking=False))
                    request = urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                    start=time.monotonic(); first=None; text=''; usage=None; done=False
                    with urllib.request.urlopen(request,timeout=240) as response:
                        for line in response:
                            if not line.startswith(b'data: '): continue
                            data=line[6:].strip()
                            if data==b'[DONE]': done=True; break
                            event=json.loads(data)
                            if event.get('usage'): usage=event['usage']
                            for choice in event.get('choices',[]):
                                fragment=choice.get('delta',{}).get('content') or ''
                                if fragment and first is None: first=time.monotonic()
                                text+=fragment
                    assert done and usage and text, (label,usage)
                    digest=hashlib.sha256(text.encode()).hexdigest(); hashes.append(digest)
                    (out/f'{label}-{temp}-{repeat}.txt').write_text(text)
                    print(json.dumps(dict(case=label,temp=temp,repeat=repeat,usage=usage,wall_s=time.monotonic()-start,first_s=first-start,sha256=digest)),flush=True)
                assert hashes[0]==hashes[1], (label,temp,'cold/cached mismatch')
        print('HOT_API_COMPLETE PASS',flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
