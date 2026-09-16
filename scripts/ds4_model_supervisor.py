#!/usr/bin/env python3
"""Tailscale-only, allowlisted model supervisor for a single ds4-server worker."""

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Profile:
    model_id: str
    context: int
    model: str
    extra: List[str]
    disk_kv: bool = True
    mtp_head: Optional[str] = None


ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "gguf"


def model_profiles() -> Dict[str, Profile]:
    # Long-context limits below are measured production capacities on the
    # M5 Max 128 GiB host, not short benchmark defaults.  DeepSeek V4 and GLM
    # completed 262K sweeps; the installed Qwen Next Q4 completed a fresh
    # 64K/128K/256K sweep with no swap on 2026-09-16.
    qwen_next = GGUF / "qwen38-q4k-tensor" / "Qwen3.8-Flash-Next-Q4KImatrixExperts-MXFP4Down-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out-MTP.gguf"
    qwen_ple = GGUF / "qwen38-q4k-tensor" / "Qwen3.8-Flash-Next-PLE-Q4_1.gguf"
    qwen_vision = GGUF / "mmproj-Qwen3.8-Flash-Next-F16.gguf"
    qwen_mtp = GGUF / "qwen-small-quality" / "mtp-Qwen3.8-27B-Q4_64A.gguf"
    glm = GGUF / "GLM-5.3-Flash-Q2-Q4K-Attention-SharedDownQ4K.gguf"
    return {
        "deepseek-v4": Profile(
            "deepseek-v4-flash", 262144,
            str(GGUF / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AOutQ4K-L27-42-chat-v2-imatrix-0731.gguf"), []),
        "deepseek-v4-vision": Profile(
            "deepseek-v4-flash", 262144,
            str(GGUF / "DeepSeek-V4-Flash-Vision-Exp-IQ2XXS-w2Q2K-AOutQ4K-L27-42.gguf"),
            ["--vision", str(GGUF / "DeepSeek-V4-Flash-Vision-Encoder.gguf")]),
        "deepseek-v41": Profile(
            "deepseek-v4.1-flash", 16384,
            str(GGUF / "DeepSeek-V4.1-Flash-IQ2XXS-w2Q2K.gguf"),
            ["--engram", str(GGUF / "DeepSeek-V4.1-Flash-Engram.gguf"),
             "--vision", str(GGUF / "DeepSeek-V4.1-Flash-Vision-Encoder.gguf"),
             "--ssd-streaming", "--ssd-streaming-cold",
             "--ssd-streaming-cache-experts", "16GB"]),
        "glm53": Profile(
            "glm-5.3-flash", 262144, str(glm),
            ["--mtp", "--mtp-exact-sampling"]),
        "glm53-vision": Profile(
            "glm-5.3-flash", 262144, str(glm),
            ["--mtp", "--mtp-exact-sampling", "--vision",
             str(GGUF / "GLM-5.3-Flash-Vision-Encoder.gguf")]),
        "qwen-next": Profile(
            "qwen3.8-flash-next", 262144, str(qwen_next),
            ["--ple", str(qwen_ple), "--mtp", "--mtp-exact-sampling"]),
        "qwen-next-vision": Profile(
            "qwen3.8-flash-next", 262144, str(qwen_next),
            ["--ple", str(qwen_ple), "--mtp", "--mtp-exact-sampling",
             "--vision", str(qwen_vision)]),
        # Dense Qwen's Metal KV pool follows the requested context.  The 35B
        # model's smaller GQA cache fits its native 262K window on 128 GB; the
        # 27B dense models use a much larger FP32 KV and are capped at the
        # allocation-tested 128K window to retain working-memory headroom.
        "qwen35": Profile(
            "qwen", 262144,
            str(GGUF / "qwen-small-quality" / "35b-mtp" / "Qwen3.6-35B-A3B-Q8_0.gguf"),
            [], disk_kv=False),
        "qwen27-q8": Profile(
            "qwen", 131072,
            str(GGUF / "qwen-small-quality" / "Qwen3.8-27B-Q8_0.gguf"),
            [], disk_kv=False, mtp_head=str(qwen_mtp)),
        "qwen27-q4": Profile(
            "qwen", 131072,
            str(GGUF / "qwen-small-quality" / "Qwen3.8-27B-Q4_64A.gguf"),
            [], disk_kv=False, mtp_head=str(qwen_mtp)),
    }


class ModelSupervisor:
    def __init__(self, host: str, api_port: int, token: str, ready_timeout: int):
        self.host = host
        self.api_port = api_port
        self.token = token
        self.ready_timeout = ready_timeout
        self.profiles = model_profiles()
        self.lock = threading.Lock()
        self.worker: Optional[subprocess.Popen] = None
        self.worker_log = None
        self.current: Optional[str] = None
        self.last_error: Optional[str] = None
        self.log_dir = Path.home() / "Library" / "Logs" / "ds4-model-supervisor"
        self.cache_dir = Path.home() / "Library" / "Caches" / "ds4-pi-kv"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _dependencies(self, profile: Profile) -> List[str]:
        paths = [profile.model]
        for index, value in enumerate(profile.extra):
            if index and profile.extra[index - 1] in ("--ple", "--engram", "--vision"):
                paths.append(value)
        if profile.mtp_head:
            paths.append(profile.mtp_head)
        return paths

    def _worker_command(self, profile: Profile) -> List[str]:
        command = [str(ROOT / "ds4-server"), "--metal", "-m", profile.model,
                   "--ctx", str(profile.context), "--host", self.host,
                   "--port", str(self.api_port)]
        if profile.disk_kv:
            command += ["--kv-disk-dir", str(self.cache_dir),
                        "--kv-disk-space-mb", "8192"]
        return command + profile.extra

    def _models(self) -> Optional[dict]:
        try:
            with self.opener.open(
                "http://{}:{}/v1/models".format(self.host, self.api_port),
                timeout=1.5,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return None

    def _healthy(self, expected_id: str) -> bool:
        if not self.worker or self.worker.poll() is not None:
            return False
        body = self._models()
        return bool(body and any(item.get("id") == expected_id for item in body.get("data", [])))

    def _stop_worker(self) -> None:
        worker = self.worker
        self.worker = None
        self.current = None
        if worker and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=10)
        if self.worker_log:
            self.worker_log.close()
            self.worker_log = None

    def _start_worker(self, profile_name: str) -> None:
        profile = self.profiles[profile_name]
        missing = [path for path in self._dependencies(profile) if not Path(path).is_file()]
        if missing:
            raise RuntimeError("missing model artifact: {}".format(missing[0]))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "{}.log".format(profile_name)
        self.worker_log = log_path.open("ab", buffering=0)
        env = dict(os.environ)
        env.pop("DS4_QWEN_MTP_HEAD", None)
        if profile.mtp_head:
            env["DS4_QWEN_MTP_HEAD"] = profile.mtp_head
        self.worker = subprocess.Popen(
            self._worker_command(profile), cwd=str(ROOT), env=env,
            stdout=self.worker_log, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.worker.poll() is not None:
                raise RuntimeError("ds4-server exited with status {}; see {}".format(
                    self.worker.returncode, log_path))
            if self._healthy(profile.model_id):
                self.current = profile_name
                self.last_error = None
                return
            time.sleep(0.25)
        raise RuntimeError("timed out loading {}; see {}".format(profile_name, log_path))

    def switch(self, profile_name: str) -> dict:
        if profile_name not in self.profiles:
            raise ValueError("unknown profile: {}".format(profile_name))
        with self.lock:
            profile = self.profiles[profile_name]
            if self.current == profile_name and self._healthy(profile.model_id):
                return self.status(reused=True)
            previous = self.current
            self._stop_worker()
            try:
                self._start_worker(profile_name)
            except Exception as error:
                self.last_error = str(error)
                self._stop_worker()
                if previous and previous != profile_name:
                    try:
                        self._start_worker(previous)
                    except Exception as rollback_error:
                        self.last_error += "; rollback failed: {}".format(rollback_error)
                raise RuntimeError(self.last_error)
            return self.status(reused=False)

    def status(self, reused: bool = False) -> dict:
        profile = self.profiles.get(self.current) if self.current else None
        return {
            "ready": bool(profile and self._healthy(profile.model_id)),
            "profile": self.current,
            "model": profile.model_id if profile else None,
            "contextWindow": profile.context if profile else None,
            "pid": self.worker.pid if self.worker and self.worker.poll() is None else None,
            "reused": reused,
            "error": self.last_error,
        }

    def close(self) -> None:
        with self.lock:
            self._stop_worker()


def handler_for(supervisor: ModelSupervisor):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, value: dict) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path not in ("/health", "/status"):
                self._send(404, {"error": "not found"})
                return
            self._send(200, supervisor.status())

        def do_POST(self) -> None:
            if self.path != "/switch":
                self._send(404, {"error": "not found"})
                return
            if self.headers.get("Authorization") != "Bearer " + supervisor.token:
                self._send(401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 8192:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send(200, supervisor.switch(str(payload.get("profile", ""))))
            except ValueError as error:
                self._send(400, {"error": str(error)})
            except Exception as error:
                self._send(500, {"error": str(error), "status": supervisor.status()})

        def log_message(self, fmt: str, *args) -> None:
            print("ds4-supervisor: " + fmt % args, flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="100.109.208.12")
    parser.add_argument("--control-port", type=int, default=8001)
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--initial", choices=sorted(model_profiles()), default="deepseek-v4")
    parser.add_argument("--token", default="dsv4-local-switch")
    parser.add_argument("--ready-timeout", type=int, default=240)
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()
    if args.list_profiles:
        for name, profile in sorted(model_profiles().items()):
            print("{}\t{}\t{}".format(name, profile.model_id, profile.context))
        return 0

    supervisor = ModelSupervisor(args.host, args.api_port, args.token, args.ready_timeout)
    server = ThreadingHTTPServer((args.host, args.control_port), handler_for(supervisor))

    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        supervisor.switch(args.initial)
        print("ds4-supervisor: control http://{}:{}; API http://{}:{}; profile={}".format(
            args.host, args.control_port, args.host, args.api_port, args.initial), flush=True)
        server.serve_forever()
    finally:
        supervisor.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
