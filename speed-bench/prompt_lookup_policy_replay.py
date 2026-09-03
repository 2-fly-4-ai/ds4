#!/usr/bin/env python3
"""Replay prompt-lookup draft policies from a ds4-agent token trace.

The trace supplies the exact prompt/checkpoint tokens and the target model's
actual continuation.  This lets candidate-selection policies be screened
without rerunning an 80-90 GiB model for every n-gram setting.  It models the
production miss/partial backoff and reports proposal coverage and accepted
tokens; finalists still require a real GPU A/B because this is not a timing
model.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Sequence


BLOCK_RE = re.compile(r"tokens label=(\S+) start=(\d+) len=(\d+)")
TOKEN_RE = re.compile(r" token index=(\d+) id=(-?\d+) bytes=")


@dataclasses.dataclass
class Trace:
    prompt: list[int]
    output: list[int]


@dataclasses.dataclass
class Result:
    name: str
    tokens: int = 0
    cycles: int = 0
    proposals: int = 0
    no_match: int = 0
    zero_accept: int = 0
    partial: int = 0
    drafted: int = 0
    accepted: int = 0
    skipped: int = 0

    def row(self) -> str:
        coverage = 100.0 * self.proposals / self.cycles if self.cycles else 0.0
        acceptance = 100.0 * self.accepted / self.drafted if self.drafted else 0.0
        tok_cycle = self.tokens / self.cycles if self.cycles else 0.0
        return (
            f"{self.name:24s} {self.cycles:6d} {self.proposals:6d} "
            f"{coverage:7.1f}% {self.drafted:7d} {self.accepted:8d} "
            f"{acceptance:7.1f}% {self.zero_accept:6d} {self.partial:7d} "
            f"{self.skipped:7d} {tok_cycle:9.3f}"
        )


def parse_trace(path: Path) -> Trace:
    lines = path.read_text(errors="replace").splitlines()
    generation_start = -1
    for i, line in enumerate(lines):
        if "prefill sync done tool_round=" in line:
            generation_start = i + 1

    prompt_by_index: dict[int, int] = {}
    raw_block_end = -1
    i = 0
    while i < len(lines):
        match = BLOCK_RE.search(lines[i])
        if not match:
            i += 1
            continue
        label, start_text, len_text = match.groups()
        start = int(start_text)
        end = int(len_text)
        expected = max(0, end - start)
        j = i + 1
        found = 0
        while j < len(lines) and found < expected:
            token = TOKEN_RE.search(lines[j])
            if not token:
                break
            index, token_id = map(int, token.groups())
            prompt_by_index[index] = token_id
            found += 1
            j += 1
        if found != expected:
            raise ValueError(
                f"{path}: incomplete {label} token block: expected {expected}, got {found}"
            )
        if label == "raw_prompt_tokens":
            raw_block_end = j
        i = j

    if generation_start < 0:
        generation_start = raw_block_end
    if generation_start < 0:
        raise ValueError(f"{path}: could not locate generation boundary")

    if not prompt_by_index:
        raise ValueError(f"{path}: trace contains no prompt token blocks")
    prompt_end = max(prompt_by_index) + 1
    missing = [idx for idx in range(prompt_end) if idx not in prompt_by_index]
    if missing:
        raise ValueError(f"{path}: prompt token indices missing near {missing[:8]}")
    prompt = [prompt_by_index[idx] for idx in range(prompt_end)]

    output: list[int] = []
    for line in lines[generation_start:]:
        token = TOKEN_RE.search(line)
        if token:
            output.append(int(token.group(2)))
    if not output:
        raise ValueError(f"{path}: trace contains no generated tokens")
    return Trace(prompt=prompt, output=output)


def fixed_candidates(
    history: Sequence[int], pending: int, ngram: int, continuation: int
) -> list[tuple[int, tuple[int, ...], int]]:
    if ngram < 1 or len(history) < ngram:
        return []
    key = list(history[-(ngram - 1) :]) + [pending] if ngram > 1 else [pending]
    candidates: list[tuple[int, tuple[int, ...], int]] = []
    for start in range(len(history) - ngram, -1, -1):
        if list(history[start : start + ngram]) != key:
            continue
        follow = start + ngram
        available = len(history) - follow
        count = min(continuation, available)
        if count > 0:
            candidates.append(
                (start, tuple(history[follow : follow + count]), available)
            )
    return candidates


def propose_current(ngram: int, consensus: int, cap: int) -> Callable:
    def propose(history: Sequence[int], pending: int) -> list[int]:
        candidates = fixed_candidates(history, pending, ngram, cap)
        eligible = [
            (pos, cont) for pos, cont, available in candidates
            if available >= consensus
        ]
        if not eligible:
            return []
        chosen = eligible[0][1]
        prefix = chosen[:consensus]
        if any(cont[:consensus] != prefix for _, cont in eligible[1:]):
            return []
        return list(chosen)

    return propose


def propose_recent(ngram: int, cap: int) -> Callable:
    def propose(history: Sequence[int], pending: int) -> list[int]:
        candidates = fixed_candidates(history, pending, ngram, cap)
        return list(candidates[0][1]) if candidates else []

    return propose


def backward_match(history: Sequence[int], pending: int, earlier_end: int) -> int:
    virtual_len = len(history) + 1
    matched = 0
    while matched <= earlier_end and matched < virtual_len:
        current_index = virtual_len - 1 - matched
        current = pending if current_index == len(history) else history[current_index]
        if current != history[earlier_end - matched]:
            break
        matched += 1
    return matched


def propose_longest(min_match: int, cap: int, consensus: int = 4) -> Callable:
    def propose(history: Sequence[int], pending: int) -> list[int]:
        matches: list[tuple[int, int, tuple[int, ...]]] = []
        for earlier_end in range(len(history) - 1, -1, -1):
            if history[earlier_end] != pending or earlier_end + 1 >= len(history):
                continue
            length = backward_match(history, pending, earlier_end)
            if length < min_match:
                continue
            cont = tuple(history[earlier_end + 1 : earlier_end + 1 + cap])
            if cont:
                matches.append((length, earlier_end, cont))
        if not matches:
            return []
        best_length = max(item[0] for item in matches)
        best = [item for item in matches if item[0] == best_length]
        if len(best) == 1:
            return list(best[0][2])
        groups = Counter(item[2][:consensus] for item in best if len(item[2]) >= consensus)
        if groups:
            winning, count = groups.most_common(1)[0]
            tied = sum(1 for value in groups.values() if value == count)
            if tied == 1:
                for _, _, cont in best:
                    if cont[:consensus] == winning:
                        return list(cont)
        return list(best[0][2])

    return propose


def propose_ranked(
    ngram: int, cap: int, signature: int = 4, dominance: float = 2.0
) -> Callable:
    def propose(history: Sequence[int], pending: int) -> list[int]:
        candidates = [
            (pos, cont)
            for pos, cont, available in fixed_candidates(
                history, pending, ngram, cap
            )
            if available >= signature
        ]
        if not candidates:
            return []
        counts = Counter(cont[:signature] for _, cont in candidates)
        ranked = counts.most_common()
        winner, win_count = ranked[0]
        other_count = sum(count for _, count in ranked[1:])
        if other_count and win_count < dominance * other_count:
            return []
        for _, cont in candidates:  # most recent member of the winning group
            if cont[:signature] == winner:
                return list(cont)
        return []

    return propose


def simulate(
    trace: Trace,
    name: str,
    propose: Callable[[Sequence[int], int], list[int]],
    backoff: bool = True,
) -> Result:
    history = list(trace.prompt)
    output = trace.output
    result = Result(name=name, tokens=len(output))
    index = 0
    skip = 0
    while index < len(output):
        result.cycles += 1
        pending = output[index]
        drafts: list[int] = []
        if skip:
            skip -= 1
            result.skipped += 1
        else:
            drafts = propose(history, pending)
        accepted = 0
        if drafts:
            result.proposals += 1
            result.drafted += len(drafts)
            while (
                accepted < len(drafts)
                and index + 1 + accepted < len(output)
                and drafts[accepted] == output[index + 1 + accepted]
            ):
                accepted += 1
            result.accepted += accepted
            if accepted == 0:
                result.zero_accept += 1
                if backoff:
                    skip = 64
            elif accepted < len(drafts):
                result.partial += 1
                if backoff:
                    skip = 32 if accepted < 4 else 8
        else:
            result.no_match += 1

        commit = 1 + accepted
        history.extend(output[index : index + commit])
        index += commit
    return result


def simulate_adaptive_depth(
    trace: Trace,
    name: str,
    shallow_cap: int,
    deep_cap: int,
    promote_streak: int,
    backoff: bool = True,
) -> Result:
    """Replay the production consensus policy with acceptance-driven depth.

    A full shallow or deep proposal increments confidence.  A no-match,
    zero-accept, or partial proposal resets it immediately.  This mirrors the
    runtime state machine while letting GLM and DeepSeek thresholds be screened
    independently.
    """
    history = list(trace.prompt)
    output = trace.output
    result = Result(name=name, tokens=len(output))
    index = 0
    skip = 0
    full_streak = 0
    shallow_propose = propose_current(24, 4, shallow_cap)
    deep_propose = propose_current(24, 4, deep_cap)
    while index < len(output):
        result.cycles += 1
        pending = output[index]
        drafts: list[int] = []
        if skip:
            skip -= 1
            result.skipped += 1
        else:
            propose = deep_propose if full_streak >= promote_streak else shallow_propose
            drafts = propose(history, pending)
        accepted = 0
        if drafts:
            result.proposals += 1
            result.drafted += len(drafts)
            while (
                accepted < len(drafts)
                and index + 1 + accepted < len(output)
                and drafts[accepted] == output[index + 1 + accepted]
            ):
                accepted += 1
            result.accepted += accepted
            if accepted == len(drafts):
                full_streak += 1
            else:
                full_streak = 0
                if accepted == 0:
                    result.zero_accept += 1
                    if backoff:
                        skip = 64
                else:
                    result.partial += 1
                    if backoff:
                        skip = 32 if accepted < 4 else 8
        else:
            result.no_match += 1
            full_streak = 0

        commit = 1 + accepted
        history.extend(output[index : index + commit])
        index += commit
    return result


def simulate_adaptive_stages(
    trace: Trace,
    name: str,
    stages: Sequence[tuple[int, int]],
    backoff: bool = True,
) -> Result:
    """Replay a cumulative full-accept streak across multiple draft depths.

    ``stages`` contains ``(minimum_streak, draft_cap)`` pairs.  The first
    stage must begin at streak zero.  A no-match or non-full proposal resets
    the policy to that first stage, exactly like the production two-stage
    controller.
    """
    if not stages or stages[0][0] != 0:
        raise ValueError("adaptive stages must begin at streak zero")
    ordered = sorted(stages)
    proposers = {
        cap: propose_current(24, 4, cap) for _, cap in ordered
    }
    history = list(trace.prompt)
    output = trace.output
    result = Result(name=name, tokens=len(output))
    index = 0
    skip = 0
    full_streak = 0
    while index < len(output):
        result.cycles += 1
        pending = output[index]
        drafts: list[int] = []
        if skip:
            skip -= 1
            result.skipped += 1
        else:
            cap = ordered[0][1]
            for threshold, stage_cap in ordered[1:]:
                if full_streak < threshold:
                    break
                cap = stage_cap
            drafts = proposers[cap](history, pending)

        accepted = 0
        if drafts:
            result.proposals += 1
            result.drafted += len(drafts)
            while (
                accepted < len(drafts)
                and index + 1 + accepted < len(output)
                and drafts[accepted] == output[index + 1 + accepted]
            ):
                accepted += 1
            result.accepted += accepted
            if accepted == len(drafts):
                full_streak += 1
            else:
                full_streak = 0
                if accepted == 0:
                    result.zero_accept += 1
                    if backoff:
                        skip = 64
                else:
                    result.partial += 1
                    if backoff:
                        skip = 32 if accepted < 4 else 8
        else:
            result.no_match += 1
            full_streak = 0

        commit = 1 + accepted
        history.extend(output[index : index + commit])
        index += commit
    return result


def policies(cap: int) -> Iterable[tuple[str, Callable, bool]]:
    yield "current-24-cons4", propose_current(24, 4, cap), True
    for ngram in (4, 8, 12, 16):
        yield f"gate-{ngram}-cons4", propose_current(ngram, 4, cap), True
    for ngram in (4, 8, 12, 16, 24):
        yield f"recent-{ngram}", propose_recent(ngram, cap), True
    for min_match in (4, 8, 12, 16, 24):
        yield f"longest-{min_match}", propose_longest(min_match, cap), True
    for ngram in (4, 8, 12, 16):
        yield f"ranked-{ngram}-2x", propose_ranked(ngram, cap), True
    yield "legacy-4-no-gate", propose_recent(4, cap), False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--cap", type=int, default=7)
    parser.add_argument(
        "--adaptive-glm",
        action="store_true",
        help="also sweep GLM depth-3 promotion policies",
    )
    args = parser.parse_args()
    if args.cap < 1:
        parser.error("--cap must be positive")

    print(
        "policy                   cycles passes coverage drafted accepted "
        "accept% zero partial skipped tok/cycle"
    )
    for path in args.trace:
        trace = parse_trace(path)
        print(f"\n# {path} prompt={len(trace.prompt)} output={len(trace.output)} cap={args.cap}")
        for name, proposer, backoff in policies(args.cap):
            print(simulate(trace, name, proposer, backoff).row())
        if args.adaptive_glm:
            for deep_cap in (5, 7, 9, 11, 15):
                for streak in (1, 2, 4, 8, 16):
                    name = f"glm-3to{deep_cap}-s{streak}"
                    print(simulate_adaptive_depth(
                        trace, name, 3, deep_cap, streak
                    ).row())
            staged = (
                ("glm-3to7-s2-to15-s8", ((0, 3), (2, 7), (8, 15))),
                ("glm-3to7-s4-to15-s8", ((0, 3), (4, 7), (8, 15))),
                ("glm-3to7-s4-to15-s12", ((0, 3), (4, 7), (12, 15))),
                ("glm-3to9-s4-to15-s8", ((0, 3), (4, 9), (8, 15))),
                ("glm-3to11-s4-to15-s8", ((0, 3), (4, 11), (8, 15))),
            )
            for name, stages in staged:
                print(simulate_adaptive_stages(trace, name, stages).row())


if __name__ == "__main__":
    main()
