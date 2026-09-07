#!/usr/bin/env python3
"""Validate recorded parity controls and print follow-up latency means."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
total = 0
for mtp in (0, 1):
    for thinking in (0, 1):
        rows = json.loads((root / f'mtp{mtp}-think{thinking}-results.json').read_text())
        assert len(rows) == 16
        total += len(rows)
        for mode in ('base', 'candidate'):
            follow = [r for r in rows if r['mode'] == mode and r['turn'] == 1]
            assert len(follow) == 4
            expected = (1980 if mode == 'candidate' else 0) if thinking else 1922
            assert all(r['usage']['prompt_tokens_details']['cached_tokens'] == expected for r in follow)
            print(f'MTP={mtp} thinking={thinking} {mode}: {mean(r["wall_s"] for r in follow)*1000:.1f} ms')
        if not thinking:
            assert len({r['content_sha256'] for r in rows if r['turn'] == 1}) == 1
    omitted = json.loads((root / f'mtp{mtp}-think1-results.json').read_text())
    replay = json.loads((root / f'mtp{mtp}-think1-replay-results.json').read_text())
    assert len(replay) == 8
    total += len(replay)
    for stream in (False, True):
        controls = [r for r in replay if r['turn'] == 1 and r['stream'] == stream]
        candidate = [r for r in omitted if r['mode'] == 'candidate' and r['turn'] == 1 and r['stream'] == stream]
        # Compare matched retained histories, not reasoning-omitted cold history.
        assert len({r['content_sha256'] for r in controls + candidate}) == 1
        assert len({r['message'].get('reasoning_content') for r in controls + candidate}) == 1
print(f'PASS: {total} API responses; cached-token and matched-history controls')
