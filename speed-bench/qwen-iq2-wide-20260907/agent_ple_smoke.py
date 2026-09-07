#!/usr/bin/env python3
"""Native-agent external PLE integration: cold bootstrap then same-recipe cache hit."""
import subprocess
from pathlib import Path
work=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
model=Path('/Users/brianfarley/Desktop/ds4/gguf/qwen38-iq2-test/Qwen3.8-Flash-Next-IQ2XXSImatrix-MXFP4Down-MTP.gguf')
ple=model.parent/'Qwen3.8-Flash-Next-PLE-Q4_1.gguf'
assert (out/'iq2-integrity.txt').exists()
for rep in range(2):
 tag=f'agent-ple-{rep}';trace=out/f'{tag}.trace'
 cmd=[str(work/'ds4-agent'),'-m',str(model),'--ple',str(ple),'--ctx','16384',
      '--non-interactive','--nothink','--temp','0','-n','32',
      '--prompt','Do not use any tools. Reply with exactly READY.', '--trace',str(trace)]
 r=subprocess.run(cmd,cwd=work,capture_output=True,text=True,timeout=180)
 (out/f'{tag}.stdout').write_text(r.stdout);(out/f'{tag}.stderr').write_text(r.stderr)
 assert r.returncode==0,(tag,r.returncode)
 assert 'READY' in r.stdout,(tag,r.stdout)
 data=trace.read_text()
 assert '/qwen-ple-' in data,(tag,'not using isolated cache')
 if rep==1:assert 'sysprompt kv hit' in data,(tag,'missing same-recipe cache reuse')
 print('PASS',tag,'READY response and isolated cache'+(' reuse' if rep else ''),flush=True)
