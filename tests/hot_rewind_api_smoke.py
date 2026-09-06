"""Focused-port regression: sequential models, explicit builds, local HTTP only.

Usage: python3 tests/verifier_port_smoke.py BASELINE CANDIDATE MODEL_ROOT OUTPUT
Hot-rewind fix validation: candidate repeats must match the baseline COLD response.
Not a statistical performance benchmark. Never changes server defaults.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

baseline, candidate, root, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
models = {
    'ds4': root / 'ds4flash.gguf',
    'qwen': root / 'gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf',
    'glm': root / 'gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf',
}
prompt = 'Write a Python function to add two integers. Briefly explain it.'
def emit(row):
    print(json.dumps(row), flush=True)
def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(('DS4_', 'WIDE_', 'ROUTER_'))}

for family, model in models.items():
    if os.environ.get('VERIFIER_SMOKE_FAMILY') and family != os.environ['VERIFIER_SMOKE_FAMILY']:
        continue
    hashes = []
    cli_runs = [] if os.environ.get('VERIFIER_SMOKE_API_ONLY') else [('baseline', baseline), ('candidate', candidate)]
    for name, build in cli_runs:
        args = [str(build / 'ds4'), '-m', str(model), '--metal', '--ctx', '2048',
                '-n', '128', '--temp', '0', '--nothink', '-p', prompt]
        env = clean_env()
        # Compare unchanged scalar execution, not the old broken GLM verifier.
        if family == 'glm': env['DS4_PROMPT_LOOKUP_DISABLE'] = '1'
        with (output / f'{family}-{name}-cli.out').open('wb') as out, (output / f'{family}-{name}-cli.log').open('wb') as log:
            subprocess.run(args, cwd=build, env=env, stdout=out, stderr=log, check=True, timeout=300)
        digest = hashlib.sha256((output / f'{family}-{name}-cli.out').read_bytes()).hexdigest()
        hashes.append(digest)
        emit(dict(test='cli', family=family, build=name, sha256=digest))
    if hashes: assert hashes[0] == hashes[1], (family, 'CLI parity')

    api_results = []
    # GLM compares the candidate router to its own scalar oracle. Other models
    # compare before/after with their existing production routes unchanged.
    runs = [('baseline', baseline), ('candidate', candidate)] if family != 'glm' else [('scalar', candidate), ('router', candidate)]
    if os.environ.get('VERIFIER_SMOKE_ABBA'): runs += runs[::-1]
    for run_id, (name, build) in enumerate(runs):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        env = clean_env()
        args = [str(build / 'ds4-server'), '-m', str(model), '--ctx', '4096', '--host', '127.0.0.1', '--port', str(port)]
        if name == 'scalar': env['DS4_PROMPT_LOOKUP_DISABLE'] = '1'
        if name == 'router': args += ['--mtp', '--mtp-exact-sampling']
        with (output / f'{family}-{run_id}-{name}-api.log').open('wb') as log:
            proc = subprocess.Popen(args, cwd=build, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 240
                while True:
                    assert proc.poll() is None, (family, name, 'server exited')
                    try:
                        with urllib.request.urlopen(base + '/v1/models', timeout=2) as r: inventory = json.load(r)
                        break
                    except (OSError, TimeoutError):
                        if time.monotonic() > deadline: raise RuntimeError('readiness timeout')
                        time.sleep(.5)
                model_id = inventory['data'][0]['id']
                cases = [('code', prompt, 128, None), ('code-repeat', prompt, 128, None),
                         ('story', 'Write a short story about a lighthouse keeper who receives a letter from tomorrow.', 128, None),
                         ('json', 'Return only a JSON array of eight sample inventory items with id, name, quantity, and price.', 128, None),
                         ('copy', (root / 'speed-bench/qwen_prompt_lookup_copy_chat.txt').read_text(), 256, None),
                         ('edit', (root / 'speed-bench/qwen_prompt_lookup_edit_chat.txt').read_text(), 256, None),
                         ('stop', prompt, 128, ['return'])]
                current = []
                for label, content, limit, stop in cases:
                    body = dict(model=model_id, messages=[dict(role='user', content=content)], temperature=0,
                                max_tokens=limit, stream=True, stream_options=dict(include_usage=True),
                                chat_template_kwargs=dict(enable_thinking=False))
                    if stop: body['stop'] = stop
                    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    start = time.monotonic(); first = None; text = ''; usage = None; done = False
                    with urllib.request.urlopen(req, timeout=180) as response:
                        for raw in response:
                            if not raw.startswith(b'data: '): continue
                            data = raw[6:].strip()
                            if data == b'[DONE]': done = True; break
                            chunk = json.loads(data)
                            if chunk.get('usage'): usage = chunk['usage']
                            for choice in chunk.get('choices', []):
                                delta = choice.get('delta', {}).get('content') or ''
                                if delta and first is None: first = time.monotonic()
                                text += delta
                    elapsed = time.monotonic() - start
                    assert done and usage and usage['completion_tokens'] > 0, (family, name, label, usage)
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    (output / f'{family}-{run_id}-{name}-{label}.txt').write_text(text)
                    row = dict(test='api', family=family, build=name, run=run_id, case=label, wall_s=elapsed,
                               first_s=first-start if first else None, usage=usage, sha256=digest)
                    emit(row); current.append(row)
                # Cold batched prefill and a cached-tail replay can already
                # differ on the baseline. Record this separately; the port
                # contract below compares each identical request/cache state
                # before/after, not two different prefill schedules.
                emit(dict(test='cold_cached_identity', family=family, build=name,
                          equal=current[0]['sha256'] == current[1]['sha256']))
                if family == 'ds4' and name == 'candidate':
                    assert current[0]['sha256'] == current[1]['sha256'], 'fixed repeat must match cold'
                api_results.append(current)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    for index, group in enumerate(zip(*api_results)):
        if family == 'ds4' and group[0]['case'] == 'code-repeat':
            expected = api_results[0][0]['sha256']
            for records in api_results:
                if records[index]['build'] == 'candidate':
                    assert records[index]['sha256'] == expected, 'repeat versus original cold'
        else:
            assert len({r['sha256'] for r in group}) == 1, (family, group[0]['case'], 'API parity')
    emit(dict(test='family_complete', family=family, status='PASS'))
emit(dict(test='complete', status='PASS'))
