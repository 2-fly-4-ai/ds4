import hashlib, json, time, urllib.request

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


