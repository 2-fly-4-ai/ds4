"""Reuse already-installed PLE bytes; accept only the exact published SHA-256."""
import hashlib
from pathlib import Path

root = Path('/Users/brianfarley/Desktop/ds4/gguf')
dest = root/'qwen38-iq2-test/Qwen3.8-Flash-Next-PLE-Q4_1.gguf'
partial = dest
source = root/'Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf'
expected = '66db3ab390f4dd5063ecc89cc180f4713898577682347001bf64ab8e328527a1'
tmp = dest.with_name(dest.name+'.verified-local')
assert not tmp.exists()
with partial.open('rb') as f:
    header = f.read(3552)
# Directory inspection found three small tensor payloads AFTER the PLE table.
# They must be included, not mistaken for additional header padding.
tail = (dest.parent/'ple-tail.bin').read_bytes()
assert len(header)==3552 and header[:4]==b'GGUF' and len(tail)==288
digest = hashlib.sha256()
with source.open('rb') as src, tmp.open('xb') as out:
    out.write(header); digest.update(header)
    src.seek(74603986880)
    remaining = 32000153600
    while remaining:
        data = src.read(min(8*1024*1024,remaining))
        assert data
        out.write(data); digest.update(data); remaining -= len(data)
    out.write(tail); digest.update(tail)
assert tmp.stat().st_size == 32000157440
actual = digest.hexdigest()
print('PLE SHA-256:', actual, flush=True)
assert actual == expected, 'Not the published sidecar; do not use this candidate'
print('Published PLE sidecar reconstructed and verified exactly; ready for promotion', flush=True)
