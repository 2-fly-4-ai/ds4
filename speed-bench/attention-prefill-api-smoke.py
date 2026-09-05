"""Local-only smoke tests; stop only the server process launched here."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

root = Path.cwd()
server_root = Path(os.environ.get("DS4_SMOKE_SERVER_ROOT", str(root)))
out = Path(os.environ.get("DS4_SMOKE_RESULT_DIR", str(root / "speed-bench/attention-prefill-results")))
models = Path("/Users/brianfarley/Desktop/ds4/gguf")
names = {
    "ds": "DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AOutQ4K-L27-42.gguf",
    "glm": "GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf",
}
cases = [
    ("code", 0, "Write a Python function that merges overlapping intervals. Include three assertions. Return only code."),
    ("code-repeat", 0, "Write a Python function that merges overlapping intervals. Include three assertions. Return only code."),
    ("story", .7, "Write a short story about a lighthouse keeper who receives a letter from tomorrow."),
    ("json", 0, "Return only a JSON array of eight sample inventory items with id, name, quantity, and price."),
]
results = []
repeat_checks = []
for model, name in names.items():
    if os.environ.get("DS4_SMOKE_MODEL") and os.environ["DS4_SMOKE_MODEL"] != model:
        continue
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with (out / f"{model}-api-server.log").open("w") as log:
        process = subprocess.Popen([str(server_root / "ds4-server"), "-m", str(models / name),
            "--ctx", "4096", "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=log,
            cwd=server_root)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{model} server exited {process.returncode}")
                try:
                    with urllib.request.urlopen(base + "/v1/models", timeout=2) as response:
                        inventory = json.load(response)
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{model} server readiness timeout")
                    time.sleep(.5)
            assert inventory.get("data"), inventory
            model_id = inventory["data"][0]["id"]
            for label, temp, prompt in cases:
                body = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
                    "temperature": temp, "max_tokens": 128, "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}}
                request = urllib.request.Request(base + "/v1/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                start = time.monotonic()
                first = None
                text = ""
                usage = None
                done = False
                with urllib.request.urlopen(request, timeout=120) as response:
                    for raw in response:
                        line = raw.decode().strip()
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            done = True
                            break
                        chunk = json.loads(payload)
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for choice in chunk.get("choices", []):
                            content = choice.get("delta", {}).get("content") or ""
                            if content and first is None:
                                first = time.monotonic()
                            text += content
                elapsed = time.monotonic() - start
                assert done and text and usage and usage["completion_tokens"] > 0, (model, label, usage)
                row = {"model": model, "case": label, "temperature": temp,
                    "wall_seconds": elapsed, "first_content_seconds": first - start,
                    "usage": usage, "end_to_end_tps": usage["completion_tokens"] / elapsed,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(), "output": text}
                results.append(row)
                print(json.dumps(row), flush=True)
            pair = [r for r in results if r["model"] == model and r["case"].startswith("code")]
            equal = pair[0]["sha256"] == pair[1]["sha256"]
            repeat_checks.append(equal)
            print(json.dumps({"model": model, "cold_cached_repeat_equal": equal}), flush=True)
            if "DS4_SMOKE_DIAGNOSTIC" not in os.environ:
                assert equal, f"{model} API repeated code mismatch"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
print(json.dumps({"status": "PASS" if all(repeat_checks) else "FAIL_REPEAT_PARITY",
    "http_requests_passed": len(results)}), flush=True)
if not all(repeat_checks):
    sys.exit(3)
