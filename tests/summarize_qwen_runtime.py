#!/usr/bin/env python3
"""Summarize matched API evidence; never count divergent output as a speed win."""
import json
import subprocess
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'speed-bench/api-health'
print('| Model / task | Prompt tokens | Off decode t/s | On decode t/s | Exact output | Repeat exact | Cached TTFT ms |')
print('|---|---:|---:|---:|---|---|---:|')
for model, prefix, label in [('35B Q8', 'qwen27-q8', '35b'),
                             ('27B Q8', 'qwen27-q8', 'port'),
                             ('27B Q4', 'qwen27-q4-64a', 'port')]:
    off, on = [root / f'{prefix}-{label}-{route}' for route in ('off', 'on')]
    for file in sorted(off.glob('[0-9][02468]-*.json')):
        peer = on / file.name
        if not peer.exists(): continue
        a, b = [json.loads(p.read_text()) for p in (file, peer)]
        repeat_file = on / f'{int(file.name[:2])+1:02d}{file.name[2:]}'
        repeat = json.loads(repeat_file.read_text()) if repeat_file.exists() else None
        def rate(r):
            n = r['usage']['completion_tokens'] - 1
            return n / max(r['wall_s'] - r['ttft_s'], 1e-6)
        same = (a['text'], a['reasoning']) == (b['text'], b['reasoning'])
        repeated = repeat and (repeat['text'], repeat['reasoning']) == (b['text'], b['reasoning'])
        print(f"| {model} {a['case']} | {a['usage']['prompt_tokens']} | {rate(a):.2f} | {rate(b):.2f} | {same} | {bool(repeated)} | {repeat['ttft_s']*1000 if repeat else 0:.1f} |")
print('\nDecode here is client-observed (completion tokens minus one)/(wall time minus TTFT). Single-run measurements, not confidence intervals. Output is capped at 256 tokens. Cold and repeat requests are paired; cache gains are not decode gains.')
repeats = []
for prefix,label in [('qwen27-q8','35b'),('qwen27-q8','port'),('qwen27-q4-64a','port')]:
    for route in ('off','on'):
        folder = root/f'{prefix}-{label}-{route}'
        for file in folder.glob('[0-9][02468]-*.json'):
            peer = folder/f'{int(file.name[:2])+1:02d}{file.name[2:]}'
            if peer.exists():
                a,b = [json.loads(p.read_text()) for p in (file,peer)]
                repeats.append((a['text'],a['reasoning'])==(b['text'],b['reasoning']) and b['usage']['prompt_tokens_details']['cached_tokens']>0)
print(f'\nCold/repeat output equality with positive cache reuse: {sum(repeats)}/{len(repeats)} pairs.')

main = Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],text=True).strip()).parent
print('\n## Regression output checks\n')
print('| Run | Completed requests | Exact against archived baseline | Median decode t/s |')
print('|---|---:|---:|---:|')
for model in ('qwen27-q8','qwen27-q4-64a','qwen-next-q4','glm53','ds4-0731','ds4-vision-exp','ds41-ssd-safe'):
    small = model.startswith('qwen27')
    current = root / f'{model}-{"port-off" if small else "port-regression"}'
    old = main / 'speed-bench/api-health' / f'{model}-{"no-mtp" if small else "regression"}'
    baseline = {json.loads(f.read_text())['case']:json.loads(f.read_text()) for f in old.glob('[0-9][0-9]-*.json')}
    records = [json.loads(f.read_text()) for f in current.glob('[0-9][0-9]-*.json')]
    if not records: continue
    pairs = [(r,baseline[r['case']]) for r in records if r['case'] in baseline]
    matched = sum((a['text'],a.get('reasoning',''))==(b['text'],b.get('reasoning','')) for a,b in pairs)
    completed = sum(r.get('done') and bool(r.get('text') or r.get('reasoning')) for r in records)
    rates = [(r['usage']['completion_tokens']-1)/max(r['wall_s']-r['ttft_s'],1e-6) for r in records if r.get('usage') and r['usage']['completion_tokens']>1]
    print(f'| {model} | {completed}/{len(records)} | {matched}/{len(pairs)} | {statistics.median(rates):.2f} |')

print('\n## Rebuilt main smoke checks\n')
for prefix,label,baseline in [('qwen27-q8','main35-verified','35b-on'),('qwen27-q8','main27-verified','port-on'),('qwen27-q4-64a','main27-verified','port-on')]:
    folder=root/f'{prefix}-{label}'
    files=[folder/f'{i:02d}-short-json.json' for i in range(2)]
    if not all(f.exists() for f in files): continue
    a,b=[json.loads(f.read_text()) for f in files]
    old=json.loads((root/f'{prefix}-{baseline}'/'06-short-json.json').read_text())
    assert a['done'] and b['done'] and a['text']==b['text']==old['text']
    assert b['usage']['prompt_tokens_details']['cached_tokens']>0
    print(f'- {prefix} {label}: exact candidate/cold/repeat output; positive cache reuse; PASS.')
