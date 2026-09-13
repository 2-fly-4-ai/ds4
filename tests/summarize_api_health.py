#!/usr/bin/env python3
import json, re, subprocess, statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'speed-bench/api-health'
lines=['# Qwen API protocol repair — validation','',
'Validation was performed in an isolated worktree before integration. No quant or kernel changes. Output caps are 256 tokens; nonempty streams alone do not certify complete answers.','',
'| Run | Complete nonempty | Leaked ChatML boundaries | JSON checks | Edit checks |','|---|---:|---:|---:|---:|']
for folder in sorted(root.iterdir()):
 if not folder.is_dir():continue
 records=[]
 for file in sorted(folder.glob('[0-9][0-9]-*.json')):
  r=json.loads(file.read_text());records.append((file,r))
 if not records:continue
 good=sum(bool(r['done'] and (r['text'].strip() or (r.get('thinking') and r['reasoning'].strip()))) for _,r in records)
 leak=sum(any(t in r['text'] for t in ('<|im_start|>','<|im_end|>','<|endoftext|>')) for _,r in records)
 js=[];ed=[]
 for _,r in records:
  if r['case'].endswith('-json'):
   try:
    v=json.loads(r['text']);js.append(isinstance(v,list) and len(v)==3 and all(isinstance(x,dict) and type(x.get('id')) is int and type(x.get('quantity')) is int and isinstance(x.get('name'),str) and type(x.get('price')) in (int,float) for x in v))
   except ValueError:js.append(False)
  if r['case'].endswith('-edit'):ed.append(bool(re.search(r'#define\s+BUFFER_CAPACITY\s+128\b',r['text']) and 'ring_push' in r['text'] and not re.search(r'\benqueue\b',r['text'])))
 lines.append(f'| {folder.name} | {good}/{len(records)} | {leak} | {sum(js)}/{len(js)} | {sum(ed)}/{len(ed)} |')
lines+=['','## MTP-on/off exact text comparison','', '| Model | Identical short greedy cases |','|---|---:|']
for model in ('qwen27-q8','qwen27-q4-64a'):
 a=root/f'{model}-final-chat';b=root/f'{model}-no-mtp'
 pairs=[]
 for f in sorted(b.glob('[0-9][0-9]-*.json')):
  other=a/f.name
  if other.exists():pairs.append(json.loads(f.read_text())['text']==json.loads(other.read_text())['text'])
 if pairs:lines.append(f'| {model} | {sum(pairs)}/{len(pairs)} |')
lines+=['','## Existing-model archived request replay','', '| Model | Identical text to main benchmark |','|---|---:|']
main=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],text=True).strip()).parent/'speed-bench/production-matrix-20260914'
for folder in sorted(root.glob('*-regression')):
 model=folder.name.removesuffix('-regression');phase='v41-16gb' if model=='ds41-ssd-safe' else 'context-sweep';pairs=[]
 config=folder/'config.json'
 if config.exists():phase=json.loads(config.read_text()).get('source_phase',phase)
 for f in folder.glob('[0-9][0-9]-*.json'):
  r=json.loads(f.read_text());old=main/phase/model/(r['case']+'.response.json')
  if old.exists():pairs.append(json.loads(old.read_text())['text']==r['text'])
 if pairs:lines.append(f'| {model} | {sum(pairs)}/{len(pairs)} |')
lines+=['','## Short-request decode sanity check','',
'Single-run medians across tasks, not a statistical speedup comparison. Model launch and prefill are excluded.','',
'| Run | Median decode t/s |','|---|---:|']
for folder in sorted(root.iterdir()):
 if not folder.is_dir() or not (folder/'server.log').exists():continue
 if not any(folder.name.endswith(s) for s in ('regression','final-chat','no-mtp')):continue
 segments=re.split(r'(?m)^.*(?:chat|completion) ctx=.* prompt start\n',(folder/'server.log').read_text(errors='replace'))[1:]
 values=[]
 for seg in segments[:5]:
  matches=re.findall(r'decoding chunk=.*?avg=([\d.]+) t/s',seg)
  if matches and 'finish=' in seg:values.append(float(matches[-1]))
 if values:lines.append(f'| {folder.name} | {statistics.median(values):.2f} |')
(root/'CHECKS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
