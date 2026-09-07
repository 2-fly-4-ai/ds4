#!/usr/bin/env python3
"""Real local API tool-turn A/B. Tools return fixture data; none are executed.

Cold and visible-KV histories differ when reasoning is omitted: record output
hashes, but do not mislabel that comparison as numerical-route equivalence.
"""
import argparse, hashlib, json, os, socket, subprocess, time, urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--rounds', type=int, default=1)
p.add_argument('--context', type=int, choices=(8192,32768), default=8192)
p.add_argument('--only-mode', choices=('base','candidate'))
p.add_argument('--mtp', action='store_true')
p.add_argument('--thinking', action='store_true')
p.add_argument('--replay-reasoning', action='store_true')
a = p.parse_args()
root = Path('/Users/brianfarley/Desktop/ds4')
candidate = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
tag = f'ctx{a.context}-mtp{int(a.mtp)}-think{int(a.thinking)}'
if a.replay_reasoning: tag += '-replay'
results = []

def request(url, body):
    start = time.perf_counter()
    req = urllib.request.Request(url+'/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    first = None
    with urllib.request.urlopen(req, timeout=180) as response:
        if not body['stream']:
            data = json.load(response)
            msg = data['choices'][0]['message']
            usage = data['usage']; finish = data['choices'][0]['finish_reason']
        else:
            msg = {'role':'assistant', 'content':''}; calls = {}; usage = {}; finish = None
            for line in response:
                if not line.startswith(b'data: '): continue
                payload = line[6:].strip()
                if payload == b'[DONE]': break
                event = json.loads(payload)
                if event.get('usage'): usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('content') or delta.get('tool_calls') or delta.get('reasoning_content'):
                        if first is None: first = time.perf_counter()-start
                    for key in ('content','reasoning_content'):
                        if delta.get(key): msg[key] = msg.get(key,'')+delta[key]
                    for call in delta.get('tool_calls', []):
                        obj = calls.setdefault(call['index'], {'id':'','type':'function','function':{'name':'','arguments':''}})
                        if call.get('id'): obj['id'] = call['id']
                        for key in ('name','arguments'):
                            obj['function'][key] += call.get('function',{}).get(key,'')
                    if choice.get('finish_reason'): finish = choice['finish_reason']
            if calls: msg['tool_calls'] = [calls[k] for k in sorted(calls)]
    return msg, {'wall_s':time.perf_counter()-start, 'ttft_s':first, 'usage':usage, 'finish':finish,
                 'content_sha256':hashlib.sha256((msg.get('content') or '').encode()).hexdigest()}

for rep in range(a.rounds):
    for mode in ((a.only_mode,) if a.only_mode else (('base','candidate') if rep%2 == 0 else ('candidate','base'))):
        work = Path('/Users/brianfarley/Desktop/ds4-qwen-mtp-port') if mode == 'base' else candidate
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
        url = f'http://127.0.0.1:{port}'
        prefix = out/f'{tag}-r{rep}-{mode}'
        cmd = [str(work/'ds4-server'), '-m',str(root/'gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf'),
               '--host','127.0.0.1','--port',str(port),'--ctx',str(a.context+4096),'--trace',str(prefix)+'.trace']
        if a.mtp: cmd += ['--mtp']
        with open(str(prefix)+'.log','w') as log:
            proc = subprocess.Popen(cmd,cwd=work,stdout=log,stderr=log)
            try:
                for _ in range(1200):
                    if proc.poll() is not None: raise RuntimeError(f'{mode} exited: {prefix}.log')
                    try:
                        with urllib.request.urlopen(url+'/v1/models',timeout=1) as r: json.load(r)
                        break
                    except OSError: time.sleep(.1)
                else: raise RuntimeError('startup timeout')
                for stream in (False, True):
                    system = 'You are a coding assistant. Read package.json, then src/main.py, then README.md using exactly one read_file call per response in that order. Wait for each result before calling the next tool. Only after all three results, summarize the project version, entry function and test command in one short sentence. Do not invent file contents.\n'
                    system += '\n'.join(f'Reference item {i}: preserve public interfaces and test changes before release.' for i in range((a.context-500)//17))
                    msgs = [{'role':'system','content':system}, {'role':'user','content':f'Case {int(stream)}: inspect the three files in the prescribed order and report the project version, entry function and test command.'}]
                    tools = [{'type':'function','function':{'name':'read_file','description':'Read a project file.','parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}}]
                    body = {'model':'qwen3.8-flash-next-chat','messages':msgs,'tools':tools,'temperature':0,'max_tokens':384 if a.thinking else 128,'stream':stream,
                            'thinking':a.thinking,'chat_template_kwargs':{'enable_thinking':a.thinking},'stream_options':{'include_usage':True}}
                    for turn in range(4):
                        msg, rec = request(url,body)
                        rec.update(mode=mode,rep=rep,stream=stream,turn=turn,mtp=a.mtp,thinking=a.thinking,message=msg)
                        results.append(rec)
                        (out/f'{tag}-results.json').write_text(json.dumps(results,indent=2))
                        print(json.dumps({k:v for k,v in rec.items() if k!='message'}),flush=True)
                        if turn < 3:
                            calls = msg.get('tool_calls',[])
                            assert rec['finish']=='tool_calls' and len(calls)==1, msg
                            assert calls[0]['function']['name']=='read_file', msg
                            expected_path = ('package.json','src/main.py','README.md')[turn]
                            assert json.loads(calls[0]['function']['arguments'])['path']==expected_path, msg
                            # Emulate a normal client omitting private reasoning.
                            keys = ('role','content','tool_calls','reasoning_content') if a.replay_reasoning else ('role','content','tool_calls')
                            msgs.append({k:v for k,v in msg.items() if k in keys})
                            msgs.append({'role':'tool','tool_call_id':calls[0]['id'],'content': ('{"name":"fixture-project","version":"2.7.4"}', 'def serve():\n    return "ready"\n', 'Run tests with pytest.')[turn]})
                        else:
                            assert all(x in (msg.get('content') or '') for x in ('2.7.4','serve','pytest')) and not msg.get('tool_calls'), msg
            finally:
                proc.terminate()
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
print('Tool contracts and follow-up answer checks: PASS',flush=True)

