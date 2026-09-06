#!/usr/bin/env python3
import subprocess, os, time, json, urllib.request, urllib.error, hashlib, socket, sys
from pathlib import Path
build=Path(__file__).resolve().parents[1]
cache_test="--cache" in sys.argv
out=build/"speed-bench/qwen-mtp-port-20260907"/("api-cache" if cache_test else "api");out.mkdir(parents=True,exist_ok=True)
cases=(("edit",256,None),("edit",128,None),("followup",128,None)) if cache_test else (("copy",256,None),("edit",256,None),("copy",128,["checksum"]),("edit",128,None))
common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],cwd=build,text=True).strip()).parent
root=Path(os.environ.get('QWEN_PORT_MODEL_ROOT',str(common)))
baseline=Path(os.environ.get('QWEN_PORT_BASELINE',str(common)))
if baseline.resolve()==build.resolve():raise SystemExit('Set QWEN_PORT_BASELINE to a separate baseline build, not the candidate itself')
sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
url=f"http://127.0.0.1:{port}"
all_results=[]
for mode in ("base","candidate"):
 env={k:v for k,v in os.environ.items() if not k.startswith(("DS4_","HOT_"))}
 env["DS4_PROMPT_LOOKUP_LOG"]="1"
 with (out/f"{mode}.log").open("xb") as log:
  cwd=baseline if mode=="base" else build
  proc=subprocess.Popen([str(cwd/"ds4-server"),"-m",str(root/"gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf"),"--host","127.0.0.1","--port",str(port),"--ctx","4096","--mtp","--mtp-exact-sampling"],cwd=cwd,env=env,stdout=log,stderr=log)
  try:
   for _ in range(3000):
    if proc.poll() is not None:raise RuntimeError("server exited")
    try:
     with urllib.request.urlopen(url+"/v1/models",timeout=1) as r:models=json.load(r)
     break
    except (OSError,urllib.error.URLError):time.sleep(.1)
   else:raise RuntimeError("server startup timed out")
   last_text=""
   for case,limit,stop in cases:
    task="edit" if case=="followup" else case
    prompt=(root/f"speed-bench/qwen_prompt_lookup_{task}_chat.txt").read_text()
    messages=[dict(role="user",content=prompt)]
    if case=="followup":messages.extend([dict(role="assistant",content=last_text),dict(role="user",content="Now explain the changes and suggest two useful tests.")])
    body=dict(model=models["data"][0]["id"],messages=messages,temperature=0,max_tokens=limit,stream=True,thinking=False,chat_template_kwargs={"enable_thinking":False},stream_options={"include_usage":True})
    if stop:body["stop"]=stop
    request=urllib.request.Request(url+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.monotonic();first=None;chunks=[];usage=None;finish=None
    with urllib.request.urlopen(request,timeout=120) as response:
     for raw in response:
      if not raw.startswith(b"data: "):continue
      data=raw[6:].strip()
      if data==b"[DONE]":break
      event=json.loads(data)
      if event.get("error"):raise RuntimeError(event["error"])
      if event.get("usage"):usage=event["usage"]
      for choice in event.get("choices",[]):
       text=choice.get("delta",{}).get("content","") or ""
       if text:
        if first is None:first=time.monotonic()
        chunks.append(text)
       if choice.get("finish_reason"):finish=choice["finish_reason"]
    if finish not in ("stop","length","tool_calls"):raise RuntimeError(f"invalid SSE finish: {finish}")
    if not usage:raise RuntimeError("missing usage")
    elapsed=time.monotonic()-t;text="".join(chunks);last_text=text
    rec=dict(mode=mode,case=case,limit=limit,stop=stop,elapsed=elapsed,ttft=first-t if first else None,usage=usage,finish=finish,sha256=hashlib.sha256(text.encode()).hexdigest())
    (out/f"{mode}-{len(all_results)}.txt").write_text(text)
    all_results.append(rec);print(json.dumps(rec),flush=True)
  finally:
   proc.terminate()
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
(out/"results.json").write_text(json.dumps(all_results,indent=2))
for a,b in zip(all_results[:len(cases)],all_results[len(cases):]):
 if a["sha256"]!=b["sha256"]:raise SystemExit("API output mismatch")
if cache_test and any(r["usage"].get("prompt_tokens_details",{}).get("cached_tokens",0)<=0 for r in all_results if r["case"]=="followup"):
 raise SystemExit("API followup did not exercise live cache reuse")
print("API streaming, stop, and subsequent request output parity PASS",flush=True)
