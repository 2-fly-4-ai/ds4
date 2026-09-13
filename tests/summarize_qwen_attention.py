"""Same-request output checks and client decode timings; never pool models."""
import json, os
from pathlib import Path
root = Path(__file__).resolve().parents[1] / 'speed-bench/api-health'
def speed(r):
    return (r['usage']['completion_tokens'] - 1) / (r['wall_s'] - r['ttft_s'])
print('| Model | Workload | Prompt tokens | Main t/s | Candidate t/s | Gain | Exact |')
print('|---|---|---:|---:|---:|---:|---|')
pairs = repeats = 0
for model in (35, 27):
    suffix = os.environ.get('QWEN_AB_SUFFIX', '')
    a = root / f'qwen27-q8-attention-{model}-base{suffix}'
    b = root / f'qwen27-q8-attention-{model}-candidate{suffix}'
    for file in sorted(a.glob('[0-9][0-9]-*.json')):
        if not (b/file.name).exists():
            continue
        old, new = json.loads(file.read_text()), json.loads((b/file.name).read_text())
        assert old['done'] and new['done']
        assert old['usage']['prompt_tokens'] == new['usage']['prompt_tokens']
        exact = (old['text'], old['reasoning'], old['finish'], old['usage']['completion_tokens']) == (new['text'], new['reasoning'], new['finish'], new['usage']['completion_tokens'])
        assert exact, (model, file)
        pairs += 1
        index = int(file.name[:2])
        if index % 2 and os.environ.get('QWEN_AB_REPEAT', '2') == '2':
            for folder in (a, b):
                prev = json.loads((folder/(f'{index-1:02d}'+file.name[2:])).read_text())
                cur = json.loads((folder/file.name).read_text())
                assert (prev['text'], prev['reasoning']) == (cur['text'], cur['reasoning'])
                assert cur['usage']['prompt_tokens_details']['cached_tokens'] > 0
                repeats += 1
            continue
        x,y=speed(old),speed(new)
        print(f'| {model}B Q8 | {old["case"]} | {old["usage"]["prompt_tokens"]} | {x:.2f} | {y:.2f} | {(y/x-1)*100:+.1f}% | yes |')
print(f'\nExact main/candidate pairs: {pairs}; exact, positively cached repeat pairs: {repeats}.')
print('MTP enabled; 256-token cap. Client decode excludes TTFT. Single cold request per cell, not confidence intervals.')
