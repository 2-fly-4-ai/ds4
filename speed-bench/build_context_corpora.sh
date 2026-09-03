#!/bin/zsh
set -euo pipefail

root=${0:A:h:h}
out_dir=${1:-"$root/speed-bench/context-corpora"}
mkdir -p "$out_dir"

# Real implementation text exercises code-shaped routing without duplicating a
# tiny snippet.  Stable lexical ordering makes the corpus reproducible.
find "$root" -maxdepth 3 -type f \( \
    -name '*.c' -o -name '*.h' -o -name '*.m' -o -name '*.metal' \
\) -not -path '*/context-corpora/*' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 awk 'FNR == 1 { print "\n/* FILE: " FILENAME " */" } { print }' \
    > "$out_dir/code.txt"

# A varied deterministic record stream: JSON objects plus SQL statements and
# free-form diagnostics.  Values change on every row to avoid a trivial highly
# repetitive speculative-decoding benchmark.
awk 'BEGIN {
    split("queued running complete failed retrying archived", states, " ")
    split("catalog billing search auth shipping analytics", services, " ")
    for (i = 1; i <= 45000; i++) {
        state = states[(i % 6) + 1]
        service = services[((i * 5) % 6) + 1]
        score = ((i * 7919) % 100000) / 1000.0
        printf("{\"event_id\":%d,\"service\":\"%s\",\"state\":\"%s\",\"score\":%.3f,\"shard\":%d,\"message\":\"request %d crossed checkpoint %d with route %d\"}\n", i, service, state, score, i % 97, i * 13, i % 4093, (i * 37) % 251)
        if (i % 5 == 0)
            printf("UPDATE task_events SET state = '\''%s'\'', score = %.3f WHERE event_id = %d AND shard = %d;\n", state, score, i, i % 97)
        if (i % 11 == 0)
            printf("SELECT service, count(*), avg(score) FROM task_events WHERE event_id BETWEEN %d AND %d GROUP BY service ORDER BY count(*) DESC;\n", i - 1000, i)
    }
}' /dev/null > "$out_dir/structured.txt"

cp "$root/speed-bench/promessi_sposi.txt" "$out_dir/prose.txt"

wc -lc "$out_dir"/*.txt
