#!/usr/bin/env python3
import subprocess, os, time, json, urllib.request, urllib.error, hashlib, socket
from pathlib import Path
out=Path("/private/tmp/qwen-promotion-api");out.mkdir(exist_ok=True)
root=Path("/Users/brianfarley/Desktop/ds4")
sock=socket.socket();sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
url=f"http://127.0.0.1:{port}"
all_results=[]
for mode in ("base","candidate"):
 env={k:v for k,v in os.environ.items() if not k.startswith(("DS4_PROMPT_LOOKUP_","DS4_QWEN_"))}
 if mode=="base":env["DS4_QWEN_PIPELINE_DISABLE"]="1"
 env["DS4_PROMPT_LOOKUP_LOG"]="1"
 with (out/f"{mode}.log").open("wb") as log:
  proc=subprocess.Popen(["./ds4-server","-m",str(root/"gguf/Qwen3.8-Flash-Next-DS4-Ivan-Q4-Compat.gguf"),"--host","127.0.0.1","--port",str(port),"--ctx","4096"],env=env,stdout=log,stderr=log)
  try:
   for _ in range(600):
    if proc.poll() is not None:raise RuntimeError("server exited")
    try:
     with urllib.request.urlopen(url+"/v1/models",timeout=1) as r:models=json.load(r)
     break
    except (OSError,urllib.error.URLError):time.sleep(.1)
   else:raise RuntimeError("server startup timed out")
   for case,limit,stop in (("copy",256,None),("edit",256,None),("copy",128,["checksum"]),("edit",128,None)):
    prompt=(root/f"speed-bench/qwen_prompt_lookup_{case}_chat.txt").read_text()
    body=dict(model="qwen3.8-flash-next-chat",messages=[dict(role="user",content=prompt)],temperature=0,max_tokens=limit,stream=True,thinking=False,chat_template_kwargs={"enable_thinking":False},stream_options={"include_usage":True})
    if stop:body["stop"]=stop
    request=urllib.request.Request(url+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.monotonic();first=None;chunks=[];usage=None;finish=None
    with urllib.request.urlopen(request,timeout=120) as response:
     for raw in response:
      if not raw.startswith(b"data: "):continue
      data=raw[6:].strip()
      if data==b"[DONE]":break
      event=json.loads(data)
      if event.get("usage"):usage=event["usage"]
      for choice in event.get("choices",[]):
       text=choice.get("delta",{}).get("content","") or ""
       if text:
        if first is None:first=time.monotonic()
        chunks.append(text)
       if choice.get("finish_reason"):finish=choice["finish_reason"]
    elapsed=time.monotonic()-t;text="".join(chunks)
    rec=dict(mode=mode,case=case,limit=limit,stop=stop,elapsed=elapsed,ttft=first-t if first else None,usage=usage,finish=finish,sha256=hashlib.sha256(text.encode()).hexdigest())
    (out/f"{mode}-{len(all_results)}.txt").write_text(text)
    all_results.append(rec);print(json.dumps(rec),flush=True)
  finally:
   proc.terminate()
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
(out/"results.json").write_text(json.dumps(all_results,indent=2))
for a,b in zip(all_results[:4],all_results[4:]):
 if a["sha256"]!=b["sha256"]:raise SystemExit("API output mismatch")
print("API streaming, stop, and subsequent request output parity PASS",flush=True)
