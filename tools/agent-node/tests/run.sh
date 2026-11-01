#!/bin/sh
# Tests of agent-node on the CPU with a small model of the same architecture as the network model
# (Qwen3.5-0.8B: hybrid, Gated DeltaNet + attention). Two nodes on this PC, clients against them:
#   1. several clients at once: forwards of different sessions in one pass, the answers as the reference's
#   2. speculative decoding (rollback of draft tokens) in two waves: the sequences of ended sessions are reused
#   3. serve: the chain and the draft model keep the start of the conversation between requests, the answers do not change
#
#   tools/agent-node/tests/run.sh build/bin/agent-node Qwen3.5-0.8B-Q8_0.gguf [WORKDIR]
set -u
B=$(realpath "$1"); M=$(realpath "$2"); W=${3:-$(mktemp -d)}; T=$(dirname "$(realpath "$0")")
mkdir -p "$W" && cd "$W" || exit 1
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
pids=""
stop() { for p in $pids; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; pids=""; }
trap stop EXIT
# start_nodes [tls]: layers 0:12 and 12: on ports 50152, 50153; the logs of the earlier start are kept (node-a.1.log, ...)
start_nodes() {
  for f in node-a.log node-b.log; do [ -f "$f" ] && mv "$f" "${f%.log}.$(date +%s%N).log"; done
  for n in a:0:12:50152 b:12::50153; do
    id=${n%%:*}; r=${n#*:}; l=${r%:*}; p=${r##*:}
    extra=""
    if [ "${1:-}" = tls ]; then
      [ -f "$id.pem" ] || "$B" identity --identity "$id.pem" > "$id.pin"
      extra="--tls --identity $id.pem"
    fi
    "$B" node -m "$M" --layers "$l" --host 127.0.0.1 --port "$p" -t 2 --seq-ctx 2048 --gpu-clocks off $extra > "node-$id.log" 2>&1 &
    pids="$pids $!"
  done
  for i in $(seq 180); do grep -q ready node-a.log && grep -q ready node-b.log && return 0; sleep 1; done
  echo "the nodes did not start"; cat node-a.log node-b.log; exit 1
}
CHAIN=127.0.0.1:50152,127.0.0.1:50153
P1="Write a short poem about the sea."
P2="Explain what a GPU is in two sentences."
P3="List three prime numbers and say why they are prime."
# f32 between the nodes: then the chain differs from the whole model only by the order of sums, the comparison is strict
GEN="-n 24 -c 1024 -t 2 --ignore-eos"
# the CPU of a test machine is slow, and three clients share it with two nodes: no failover on a slow step
CLI="--timeout 30000"
i=0
for P in "$P1" "$P2" "$P3"; do
  i=$((i + 1))
  "$B" reference -m "$M" -p "$P" $GEN --dump "ref$i.txt" > "ref$i.log" 2>&1 || { echo "reference failed"; cat "ref$i.log"; exit 1; }
done

echo "== 1. three clients at once"
start_nodes
cp=""
i=0
for P in "$P1" "$P2" "$P3"; do
  i=$((i + 1))
  "$B" client -m "$M" --chain $CHAIN $CLI -p "$P" $GEN --hidden f32 --force "ref$i.txt" --dump "batch$i.txt" > "batch$i.log" 2>&1 &
  cp="$cp $!"
done
wait $cp
for i in 1 2 3; do
  check "client $i follows the reference" "\"$B\" compare batch$i.txt ref$i.txt --tokens 24 > cmp-batch$i.log 2>&1"
done
shared=$(grep -h "in passes with other sessions" node-b.log | sed -E 's/.* ([0-9]+) of them in passes.*/\1/' | awk '{s += $1} END {print s + 0}')
check "forwards of different sessions ran in one pass ($shared)" "[ $shared -gt 0 ]"

echo "== 2. speculative decoding, two waves"
for w in 1 2; do
  cp=""
  for i in 2 3; do
    eval "P=\$P$i"
    "$B" client -m "$M" --chain $CHAIN $CLI -p "$P" $GEN --hidden f32 --draft "$M" --draft-max 4 --draft-noise 0.5 --dump "spec$w$i.txt" > "spec$w$i.log" 2>&1 &
    cp="$cp $!"
  done
  wait $cp
  for i in 2 3; do
    check "wave $w client $i: drafts rejected and rolled back" "grep -qE 'speculative: .* [0-9]+ of [0-9]+ draft tokens accepted' spec$w$i.log && ! grep -q 'error' spec$w$i.log"
    check "wave $w client $i writes the reference's text" "\"$B\" compare spec$w$i.txt ref$i.txt --tokens 24 > cmp-spec$w$i.log 2>&1"
  done
done
grep -h '^error' node-a.log node-b.log
check "no errors on the nodes" "! grep -q '^error' node-a.log node-b.log"
stop

echo "== 3. serve keeps the start of the conversation"
start_nodes tls
python3 "$T/fake_coordinator.py" 18080 "127.0.0.1:50152=$(cat a.pin)" "127.0.0.1:50153=$(cat b.pin)" > coord.log 2>&1 &
pids="$pids $!"
for keep in 180 0; do
  "$B" serve -m "$M" --coordinator http://127.0.0.1:18080 --port 8090 -c 4096 -t 2 --account "acc$keep.txt" \
    --keep-chain-s $keep --draft "$M" --draft-max 4 > "serve$keep.log" 2>&1 &
  sp=$!
  for i in $(seq 60); do grep -q "serve ready" "serve$keep.log" && break; sleep 1; done
  python3 "$T/agent_steps.py" 8090 > "steps$keep.jsonl" 2>&1
  kill $sp; wait $sp 2>/dev/null
done
cat steps180.jsonl
grep -h "serve:" serve180.log
cached() { sed -n "${1}p" steps180.jsonl | python3 -c 'import json, sys; print(json.load(sys.stdin)["cached"])'; }
check "the next step reads only the new text ($(cached 2) cached)" "[ \"$(cached 2)\" -gt 0 ]"
check "a new conversation keeps the system message ($(cached 3) cached)" "[ \"$(cached 3)\" -gt 0 ]"
check "the draft model keeps its start too" "grep -E 'serve: .*, [1-9][0-9]* in the draft model' serve180.log | grep -q ."
answers() { python3 -c 'import json, sys; [print(json.dumps([d["content"], d["tool_calls"]])) for d in map(json.loads, open(sys.argv[1]))]' "$1"; }
answers steps180.jsonl > answers180.txt
answers steps0.jsonl > answers0.txt
check "the answers are the same as without the cache" "[ \$(wc -l < answers180.txt) -eq 3 ] && cmp -s answers180.txt answers0.txt"
check "the reports of a session are a running total" "! grep -q ERROR coord.log && grep -q report coord.log"

[ $fail -eq 0 ] && echo "all tests passed" || { echo "logs: $W"; exit 1; }
