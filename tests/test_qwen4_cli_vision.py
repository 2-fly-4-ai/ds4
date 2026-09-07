#!/usr/bin/env python3
"""Model-backed regression test for Qwen CLI image turns and follow-up text.

Run with uv run tests/test_qwen4_cli_vision.py --model MAIN --ple PLE
--vision MMPROJ --image FIRST.png --image SECOND.png. Tests ordinary and MTP
decode. Use two different images to exercise image changes in one conversation.
"""

import argparse
import pathlib
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./ds4")
    parser.add_argument("--model", required=True)
    parser.add_argument("--ple")
    parser.add_argument("--vision", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--ordinary-only", action="store_true",
                        help="shared CLI regression on models without embedded MTP")
    args = parser.parse_args()
    if len(args.image) != 2:
        parser.error("provide two --image arguments")
    images = [pathlib.Path(path).resolve(strict=True) for path in args.image]
    if any("\n" in str(path) or "\r" in str(path) for path in images):
        parser.error("image paths must not contain line breaks")
    out = pathlib.Path(args.out_dir or tempfile.mkdtemp(prefix="qwen-cli-vision-"))
    out.mkdir(parents=True, exist_ok=True)
    commands = "".join(
        f"/read {path}\n"
        "Describe only the most recent image, including any text and colors.\n"
        for path in images
    ) + "/quit\n"
    base = [str(pathlib.Path(args.binary).resolve()), "-m", args.model,
            "--vision", args.vision, "--ctx", "8192",
            "--temp", "0", "--nothink", "-n", "180"]
    if args.ple: base += ["--ple", args.ple]
    print(f"Logs: {out}", flush=True)
    modes = [("ordinary", [])]
    if not args.ordinary_only:
        modes.append(("mtp", ["--mtp"]))
    for mode, extra in modes:
        result = subprocess.run(base + extra, input=commands, text=True,
                                capture_output=True, timeout=600)
        (out / f"{mode}.stdout").write_text(result.stdout)
        (out / f"{mode}.stderr").write_text(result.stderr)
        # The REPL can return zero after an image turn fails if a later text
        # turn succeeds. Check diagnostics as well as the process exit code.
        errors = re.findall(r"^.*(?:failed|outside the vocabulary).*$",
                            result.stderr, re.MULTILINE | re.IGNORECASE)
        encoded = len(re.findall(r"ds4: image \d+x\d+, \d+ image tokens",
                                 result.stderr))
        completed = result.stderr.count("ds4: prefill:")
        if result.returncode or errors or encoded != 2 or completed != 4:
            raise SystemExit(
                f"FAIL {mode}: exit={result.returncode}, encoded={encoded}/2, "
                f"completed={completed}/4, errors={errors}; see {out}")
        print(f"PASS {mode}: both image turns and both text follow-ups completed",
              flush=True)


if __name__ == "__main__":
    main()
