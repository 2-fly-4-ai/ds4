#!/usr/bin/env python3
"""Serial fixed-token streaming validation; no physical-cold-disk claims."""
import argparse, csv, hashlib, json, os, subprocess
from pathlib import Path

out = Path(__file__).resolve().parent
work = out.parents[1]
main = Path('/Users/brianfarley/Desktop/ds4')
prompt = main/'speed-bench/qwen-iq2-wide-20260907/fixed-corpus.txt'
models = {'glm': main/'gguf/GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf',
          'ds4': main/'ds4flash.gguf'}
parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=['parity', 'timing', 'state', 'diagnose', 'aligned'])
args = parser.parse_args()
results = []

def save(row):
    results.append(row)
    (out/f'{args.phase}-results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(row), flush=True)

def bench(tag, model, root, ctx, count, cache=None, dump=False, disable=False, alloc=None):
    cmd = [str(root/'ds4-bench'), '-m', str(models[model]), '--prompt-file', str(prompt),
           '--ctx-start', str(ctx), '--ctx-max', str(ctx), '--ctx-alloc', str(alloc or ctx+2048),
           '--gen-tokens', str(count), '--teacher-forced-decode', '--csv', str(out/f'{tag}.csv')]
    if cache:
        cmd += ['--ssd-streaming', '--ssd-streaming-cache-experts', f'{cache}GB']
    logdir = out/f'{tag}-logits'
    if dump:
        logdir.mkdir(exist_ok=True)
        cmd += ['--dump-decode-logits-dir', str(logdir)]
    env = os.environ.copy()
    if disable:
        env['DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE'] = '1'
    print('START', tag, flush=True)
    with (out/f'{tag}.log').open('w') as log:
        p = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=log, timeout=900)
    row = {'tag': tag, 'command': cmd, 'exit': p.returncode}
    if p.returncode == 0:
        row['csv'] = list(csv.DictReader((out/f'{tag}.csv').open()))
        if dump:
            row['hashes'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(logdir.glob('*.f32'))}
            assert len(row['hashes']) == count
    save(row)
    if p.returncode: raise SystemExit(f'{tag} failed')
    return row

if args.phase == 'aligned':
    # At 64K allocation BOTH modes use the existing 4K attention/work cap.
    # Actual prompt remains 8K; no 64K prefill is performed.
    ref = bench('glm-8192-aligned-ref', 'glm', main, 8192, 64, dump=True, alloc=65536)
    for cache in [8, 32]:
        row = bench(f'glm-8192-aligned-cache{cache}', 'glm', work, 8192, 64,
                    cache, True, alloc=65536)
        assert row['hashes'] == ref['hashes'], 'aligned logits mismatch'
    ref = bench('ds4-2048-resident-ref', 'ds4', main, 2048, 64, dump=True)
    row = bench('ds4-2048-cache8', 'ds4', work, 2048, 64, 8, True)
    assert row['hashes'] == ref['hashes'], 'DeepSeek streaming mismatch'
elif args.phase == 'diagnose':
    bench('glm-8192-old-fallback', 'glm', main, 8192, 64, 8, True)
    bench('glm-8192-disabled', 'glm', work, 8192, 64, 8, True, True)
elif args.phase == 'parity':
    for model, ctx, count in [('glm', 2048, 64), ('glm', 8192, 64), ('ds4', 2048, 64)]:
        # At this allocation the old GLM streaming attention/work cap differs
        # from resident mode. Compare like-for-like; phase aligned additionally
        # checks resident parity with matching caps, without altering policy.
        old_stream = model == 'glm' and ctx == 8192
        ref = bench(f'{model}-{ctx}-'+('old-fallback' if old_stream else 'resident-ref'),
                    model, main, ctx, count, 8 if old_stream else None, True)
        for cache in ([8, 32] if model == 'glm' else [8]):
            row = bench(f'{model}-{ctx}-cache{cache}', model, work, ctx, count, cache, True)
            assert row['hashes'] == ref['hashes'], f'{row["tag"]} LOGIT MISMATCH'
        if model == 'glm' and ctx == 2048:
            row = bench('glm-disabled-control', model, work, ctx, count, 8, True, True)
            assert row['hashes'] == ref['hashes'], 'disabled control mismatch'
elif args.phase == 'timing':
    # ABBA, same token sequence, without logit writes in timing windows.
    for i, root in enumerate([main, work, work, main]):
        bench(f'glm-stream-abba-{i}', 'glm', root, 2048, 256, 8)
    for cache in [16, 32]:
        bench(f'glm-stream-budget{cache}', 'glm', work, 2048, 256, cache)
    for model in ['glm', 'ds4']:
        for i, root in enumerate([main, work, work, main]):
            bench(f'{model}-resident-abba-{i}', model, root, 2048, 256)
else:
    for mtp in [0, 1]:
        env = os.environ.copy()
        env.update(DS4_TEST_MODEL=str(models['glm']), DS4_TEST_SSD_STREAMING='1',
                   DS4_TEST_SSD_STREAMING_CACHE_GB='8', DS4_TEST_GLM_MTP=str(mtp))
        tag = f'glm-stream-state-mtp{mtp}'
        with (out/f'{tag}.log').open('w') as log:
            p = subprocess.run([str(work/'ds4_test'), '--session-snapshot', '--session-rewind'],
                               cwd=work, env=env, stdout=log, stderr=log, timeout=600)
        save({'tag': tag, 'exit': p.returncode})
        if p.returncode: raise SystemExit(tag+' failed')
