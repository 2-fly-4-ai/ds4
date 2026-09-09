#!/usr/bin/env python3
"""Matched Qwen dense API prefill/decode benchmark.

Runs each configuration in a fresh server process, records local wall/TTFT,
usage, and output hashes, and rejects any output mismatch against scalar
prefill.  Intended for focused experiments; it never changes defaults.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.request


if len(sys.argv) != 4:
    raise SystemExit(f"usage: {sys.argv[0]} BUILD MODEL OUTPUT_DIR")

build, model, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
server_binary = Path(os.environ.get("QWEN_MATRIX_SERVER", build / "ds4-server"))
batch_cap = os.environ.get("QWEN_MATRIX_BATCH_CAP", "")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith("DS4_")}


def request(base, prompt, max_tokens, temperature, seed):
    body = {
        "model": "qwen",
        "prompt": prompt,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    started = time.monotonic()
    first = None
    text = ""
    usage = None
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("text") or ""
                if piece and first is None:
                    first = time.monotonic()
                text += piece
    ended = time.monotonic()
    if not usage:
        raise RuntimeError("stream did not return usage")
    return {
        "wall_s": ended - started,
        "ttft_s": first - started if first else None,
        "usage": usage,
        "text": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


seed = (
    "A storage engine maintains versioned pages in a write-ahead log. "
    "Readers pin a snapshot while writers append records, validate checksums, "
    "and atomically publish a new root. Recovery replays only committed epochs. "
)
cases = [
    ("code-short", "Write a Python function that merges overlapping intervals and explain its complexity.", 48, 0.0, 1),
    ("code-longgen", "Write a complete Python implementation of an LRU cache with type hints, tests, and a complexity discussion.", 256, 0.0, 1),
    ("creative-sampled", "Write a vivid opening paragraph about a city whose clocks begin running backward.", 64, 0.7, 987654321),
    ("systems-medium", (seed * 8) + "\nSummarize the design and identify two failure modes.", 48, 0.0, 1),
    ("systems-medium-longgen", (seed * 8) + "\nWrite a detailed design review with concrete repairs.", 256, 0.0, 1),
    ("systems-long", (seed * 24) + "\nSummarize the design and identify two failure modes.", 32, 0.0, 1),
]
configs = [
    ("scalar-a", False, False, False),
    ("batch-a", True, False, False),
    ("batch-mtp", True, True, False),
    ("batch-mtp-warm", True, True, True),
    ("batch-mtp-warm-b", True, True, True),
    ("batch-mtp-b", True, True, False),
    ("batch-b", True, False, False),
    ("scalar-b", False, False, False),
]
case_filter = {x for x in os.environ.get("QWEN_MATRIX_CASES", "").split(",") if x}
config_filter = {x for x in os.environ.get("QWEN_MATRIX_CONFIGS", "").split(",") if x}
if case_filter:
    cases = [case for case in cases if case[0] in case_filter]
if config_filter:
    configs = [config for config in configs if config[0] in config_filter]
if not cases or not configs:
    raise RuntimeError("benchmark filters selected no cases or configurations")
rows = []
for config, batched, mtp, mtp_prefill in configs:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = clean_env()
    env["DS4_QWEN_PREFILL_BATCH"] = "1" if batched else "0"
    if batch_cap:
        env["DS4_QWEN_PREFILL_BATCH_CAP"] = batch_cap
    if mtp:
        env["DS4_QWEN_MTP_K"] = "4"
        env["DS4_QWEN_MTP_PROFILE"] = "1"
    else:
        env["DS4_MTP_SPEC_DISABLE"] = "1"
    env["DS4_QWEN_MTP_PREFILL"] = "1" if mtp_prefill else "0"
    args = [str(server_binary), "-m", str(model), "--metal",
            "--ctx", "4096", "--tokens", "64", "--host", "127.0.0.1",
            "--port", str(port)]
    log_path = output / f"{config}.log"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(args, cwd=build, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 240
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"{config} server exited during startup")
                try:
                    with urllib.request.urlopen(base + "/v1/models", timeout=2):
                        break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{config} readiness timeout")
                    time.sleep(.25)
            for label, prompt, limit, temperature, seed_value in cases:
                result = request(base, prompt, limit, temperature, seed_value)
                text = result.pop("text")
                (output / f"{config}-{label}.txt").write_text(text)
                row = {"config": config, "case": label, **result}
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

for label, *_ in cases:
    group = [row for row in rows if row["case"] == label]
    hashes = {row["sha256"] for row in group}
    if len(hashes) != 1:
        raise AssertionError((label, "output parity", group))

for row in rows:
    log_text = (output / f"{row['config']}.log").read_text(errors="replace")
    # Multiple cases share one log. Preserve raw logs as the timing oracle;
    # local wall and TTFT remain unambiguous in the JSONL output.
    row["server_log_present"] = bool(re.search(r"prompt done|prefill", log_text))

(output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps({"status": "PASS", "rows": len(rows)}), flush=True)
