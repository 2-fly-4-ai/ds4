"""Summarize this focused port without turning A/A noise into speedup claims."""
import csv
import re
from pathlib import Path

root = Path(__file__).resolve().parent / "attention-prefill-results"
with (root / "prefill.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["corpus", "input_tokens", "old_tps", "new_tps", "gain_percent", "saved_ms", "exact_runs"])
    for p in (root / f"ds-prefill-{kind}-1024.log" for kind in ("code", "structured", "prose")):
        s = p.read_text()
        rates = dict(re.findall(r"aggregate variant=(control|candidate).*tokens_per_second=([\d.]+)", s))
        exact = re.search(r"exact_runs=(\d+)", s)
        assert len(rates) == 2 and exact and int(exact[1]) == 16, p
        # The rollback flag makes harness 'candidate' the OLD implementation.
        old, new = float(rates["candidate"]), float(rates["control"])
        w.writerow([p.stem.split("-")[2], 1024, old, new, (new/old-1)*100,
                    1024*(1/old-1/new)*1000, exact[1]])

with (root / "generation.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["model", "settings", "case", "input_tokens", "runs", "output_tokens",
                "generation_tps", "prefill_ms_mean", "plain_tokens", "lookup_tokens", "neural_tokens", "exact"])
    for filename, expected_cases in (("ds-greedy.log", 5), ("glm-greedy.log", 5),
                                     ("glm-sampled.log", 4), ("glm-mtp.log", 2)):
        p = root / filename
        groups = {}
        for line in p.read_text().splitlines():
            if not line.startswith("case="):
                continue
            row = dict(re.findall(r"(\w+)=([^ ]+)", line))
            if int(row["run"]) <= 0:
                continue
            groups.setdefault(Path(row["case"]).name, []).append(row)
        assert len(groups) == expected_cases, (p, len(groups))
        for name, rows in groups.items():
            exact = len(rows) == 4 and all(r["token_exact"] == r["logits_exact"] == "1" for r in rows)
            assert exact, (p, name)
            output = sum(int(r["output"]) for r in rows)
            elapsed = sum(float(r["gen_s"]) for r in rows)
            counts = [sum(int(r[k]) for r in rows) for k in ("plain_n", "lookup_n", "mtp_n")]
            w.writerow([p.stem.split("-")[0], p.stem.split("-")[1], name, rows[0]["input"],
                        len(rows), output, output/elapsed,
                        1000*sum(float(r["prefill_s"]) for r in rows)/len(rows), *counts, exact])

for n in (512, 1024, 1025, 2048):
    p = root / f"ds-activation-{n}.log"
    hits = p.read_text().count("ds4: short-prefill tail32")
    assert (hits > 0) == (n == 1024), (p, hits)
for p in (root / "ds-state-1024.log", root / "glm-state-2048.log", root / "glm-state-8192.log"):
    assert "exact_rows=257 " in p.read_text(), p
with (root / "barrier-timing.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["prefix_tokens", "old_tps", "new_tps", "timing_change_percent", "input_hash", "note"])
    for n in (2048, 8192):
        rows = []
        for p in sorted(root.glob(f"glm-barrier-{n}-*.log")):
            lines = [s for s in p.read_text().splitlines() if s.startswith("label=")]
            assert len(lines) == 1, p
            rows.append(dict(re.findall(r"(\w+)=([^ ]+)", lines[0])))
        assert len(rows) == 4 and len({r["input_hash"] for r in rows}) == 1, (n, rows)
        rates = {}
        for label in ("old", "new"):
            pair = [r for r in rows if r["label"] == label]
            assert len(pair) == 2
            rates[label] = sum(int(r["tokens"]) for r in pair) / sum(float(r["seconds"]) for r in pair)
        w.writerow([n, rates["old"], rates["new"], (rates["new"]/rates["old"]-1)*100,
                    rows[0]["input_hash"], "fixed-token timing only; old kernel has known correctness fault"])
print("PASS: prefill, route repeatability, state parity, activation and fixed-input timing gates; CSVs written.")
