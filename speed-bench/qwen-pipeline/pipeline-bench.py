#!/usr/bin/env python3
import argparse, subprocess, os, re, time, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--modes", default="warmup,base,final,final,base")
p.add_argument("--cases", default="copy,edit")
p.add_argument("--tokens",type=int,default=256)
p.add_argument("--mtp",action="store_true")
p.add_argument("--out",required=True)
a=p.parse_args()
root=Path("/Users/brianfarley/Desktop/ds4")
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
prompts={
"copy":root/"speed-bench/qwen_prompt_lookup_copy_chat.txt",
"edit":root/"speed-bench/qwen_prompt_lookup_edit_chat.txt",
"long":Path("/private/tmp/qwen-30k-profile.txt"),
"code":out/"code.txt","story":out/"story.txt","json":out/"json.txt","records":out/"records.txt","copy8k":out/"copy8k.txt","edit8k":out/"edit8k.txt"}
prompts["lateedit"]=out/"lateedit.txt"
prompts["lateedit"].write_text(prompts["copy"].read_text().replace("copied verbatim eight times", "copied eight times, with one exception: in the fifth copy only, replace the final word updates with maintenance; all other copies must remain verbatim"))
background="\n".join(f"Archive reference {i:04d}: maintain a chronological record of access requests, verify the checksum before publication, and preserve the recovery marker." for i in range(300))
for task in ("copy","edit"):
 prompts[task+"8k"].write_text("Background reference only:\n"+background+"\n\nActual task:\n"+prompts[task].read_text())
records_text="\n".join(f"item_{i:03d},zone_{i%7},enabled,units_{i*7:04d},priority_normal,retain_until_review,checksum_{i*31:05d}" for i in range(1,41))
prompts["records"].write_text("Return exactly one edited copy of the CSV below. Replace enabled with disabled in every row, preserving every other character. No Markdown, headings or explanation. Two identical reference copies follow.\n\n"+records_text+"\n\n"+records_text)
for case,txt in {
"code":"Write a Python function that detects a cycle in a directed graph and returns one actual cycle. Handle disconnected graphs. Include tests and explain complexity.",
"story":"Write a vivid original story about a retired cartographer who discovers a city that moves every night. Use dialogue and an unexpected but coherent ending.",
"json":"Return only a JSON array of 12 distinct test cases for a calendar booking API. Each object needs an id, scenario, input, expected_status and reason. Include concurrency, timezone, and validation cases."
}.items(): prompts[case].write_text(txt)
records=[]
for case in a.cases.split(","):
 for mode in a.modes.split(","):
  idx=len(records)+1
  if mode not in ("warmup", "base", "final"):
   raise SystemExit("Supported modes: warmup,base,final")
  env={k:v for k,v in os.environ.items() if not k.startswith(("DS4_PROMPT_LOOKUP_","DS4_QWEN_"))}
  env["DS4_PROMPT_LOOKUP_LOG"]="1"
  if mode in ("warmup", "base"):
   env["DS4_QWEN_PIPELINE_DISABLE"]="1"
  # final is the normal production default: no enabling flags.
  prefix=out/f"{idx:02}-{case}-{mode}"
  cmd=["./ds4","-m",str(root/"gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf"),"--metal","-c","32768" if case=="long" or case.endswith("8k") else "4096","-n",str(a.tokens),"--temp","0","--nothink","--prompt-file",str(prompts[case])]
  if a.mtp: cmd.append("--mtp-timing")
  t=time.monotonic()
  with prefix.with_suffix(".out").open("wb") as stdout, prefix.with_suffix(".log").open("wb") as stderr:
   r=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr)
  elapsed=time.monotonic()-t
  log=prefix.with_suffix(".log").read_text(errors="replace")
  match=re.search(r"prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s",log)
  counts=re.findall(r"drafted=(\d+) committed=(\d+) verify=([\d.]+) ms",log)
  rec=dict(case=case,mode=mode,exit=r.returncode,elapsed=elapsed,prefill=float(match[1]) if match else None,tps=float(match[2]) if match else None,sha256=hashlib.sha256(prefix.with_suffix(".out").read_bytes()).hexdigest(),drafted=sum(int(x[0]) for x in counts),committed=sum(int(x[1]) for x in counts),passes=len(counts),max_draft=max([int(x[0]) for x in counts],default=0))
  records.append(rec);print(json.dumps(rec),flush=True)
  (out/"results.json").write_text(json.dumps(records,indent=2))
  if r.returncode:raise SystemExit("benchmark failed; inspect logs")
