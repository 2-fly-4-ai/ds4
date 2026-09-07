#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean

p=Path(__file__).resolve().parent
for ctx in (8192,32768):
 for mtp in (0,1):
  tag=f'ctx{ctx}-mtp{mtp}-think1'
  omitted=json.loads((p/f'{tag}-results.json').read_text())
  replay=json.loads((p/f'{tag}-replay-results.json').read_text())
  if ctx==8192 and mtp==1:
   print(tag, 'FAILED tool contract; excluded from successful latency/parity matrix.',
         'Captured responses:',len(omitted),len(replay))
   continue
  assert len(omitted)==16 and len(replay)==16
  for stream in (False,True):
   for turn in range(4):
    # Same retained history, same stream mode and MTP setting. Tool IDs vary.
    refs=[r for r in replay if r['stream']==stream and r['turn']==turn]
    actual=[r for r in omitted if r['mode']=='candidate' and r['stream']==stream and r['turn']==turn]
    rows=refs+actual
    assert len(rows)==3
    assert len({r['content_sha256'] for r in rows})==1,(tag,stream,turn,'content')
    assert len({r['message'].get('reasoning_content') for r in rows})==1,(tag,stream,turn,'reasoning')
    funcs=[json.dumps([c['function'] for c in r['message'].get('tool_calls',[])],sort_keys=True) for r in rows]
    assert len(set(funcs))==1,(tag,stream,turn,'tool')
  for mode in ('base','candidate'):
   follow=[r for r in omitted if r['mode']==mode and r['turn']>0]
   print(tag,mode,'mean followup seconds',round(mean(r['wall_s'] for r in follow),4),
         'new prompt tokens',[r['usage']['prompt_tokens']-r['usage']['prompt_tokens_details']['cached_tokens'] for r in follow])
print('PASS: 96 responses in three completed A/B/control pairs; retained-history content, reasoning and tool controls match. 8K MTP tool-contract failure excluded, not passed.')
