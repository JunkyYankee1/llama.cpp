# agent-node

One part of a model in a chain of nodes, and the client that drives the chain.

```
client --tokens--> node A [layers 0..a) --int8 hidden--> node B [a..b) --> ... --> node K [k..n) --top-k logits--> client
```

Every node runs a layer range of the model (`llama_model_params.layer_start/layer_end`) and loads only the
weights of its layers. Per token each node receives one frame and sends one frame: the hidden state of the token
as ggml Q8_0 blocks (`34/32 * n_embd` bytes plus a 27-byte header; `--hidden f16` or `f32` on request).
The last node returns the top-k logits to the client.

## Usage

```sh
# three nodes on one machine, a 4-layer model split 0:1, 1:3, 3:end
agent-node node -m model.gguf --layers 0:1 --port 50152
agent-node node -m model.gguf --layers 1:3 --port 50153
agent-node node -m model.gguf --layers 3:  --port 50154

# the client loads only the tokenizer of the same model file
agent-node client -m model.gguf --chain 127.0.0.1:50152,127.0.0.1:50153,127.0.0.1:50154 \
    -p "Hello" -n 128 --ignore-eos --dump chain.txt --json chain.json

# the same generation in one process, then compare the two
agent-node reference -m model.gguf -p "Hello" -n 128 --ignore-eos --dump ref.txt
agent-node compare chain.txt ref.txt --tokens 128
```

Speculative decoding: a small draft model of the same family on the client proposes `--draft-max` tokens, one pass
over the chain checks them all, the longest matching prefix is accepted plus the next token of the chain. The text
is the greedy text of the chain alone; the cost of the network is paid once per pass instead of once per token.

```sh
agent-node client -m model.gguf --chain ... --draft draft.gguf --draft-max 6
```

The nodes undo rejected draft tokens by themselves: a forward that starts before the end of the previous one rolls
the memory back (`llama_memory_seq_rm`; the recurrent layers keep `n_rs_seq` snapshots, set from `n_rollback` of
`open`). The draft model saves its recurrent state before every draft (`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`) and
restores it after the check.

`--json` writes the speed of the prompt and of the generation and, for every node, the bytes in and out and the
compute time per generated token.

## Protocol

Frames are `u32 body size | u8 type | body`, little-endian, see `protocol.h`:

| Frame | Direction | Body |
| --- | --- | --- |
| `open` | client -> node | version, session, n_ctx, n_top, address of the next node (empty for the last node) |
| `open_ok` | node -> client | layer range, n_layer, n_embd, n_vocab, model description |
| `attach` | node -> next node | session: the following `forward` frames come over this connection |
| `forward` | client -> first node, node -> next node | session, step, pos0, n_tokens, payload kind, flags, tokens or fp16 rows |
| `result` | last node -> client | session, step, top-k (id, logit) per output token |
| `stats` / `stats_ok` | client <-> node | bytes in and out, tokens, compute time of the session |
| `close` | client -> node | session |
| `error` | any | message; errors of a next node are relayed to the client |

The client opens the session from the last node to the first one, so that every node can attach to the
already open session of its next node. The client checks that the layer ranges cover the model without gaps.

## Sessions on one node

A node keeps one context for all its sessions, a sequence of it for each. The forwards that wait at the node
at the moment (of different sessions) run in one pass, so the weights of the part are read once for them all:
on an RTX 5060 Ti four sessions get 2.7 times the tokens per second of one. Nothing waits for a pass to fill up:
a pass takes what came while the previous one ran.

- `--max-sessions 4` sequences of `--seq-ctx 16384` tokens; a session asks for its `n_ctx` out of them.
- `--seq-rollback 6`: how many draft tokens a session may roll back (`n_rollback` of `open`, the `--draft-max` of the client).
- On a GPU the node measures the memory of a session (KV cache, DeltaNet states and their snapshots) and takes
  fewer sessions if they would not fit with `--vram-margin-mb 256` to spare; the coordinator gets the number in `max_sessions`.
- `--batch-max N` forwards in a pass at most; `1` runs them one at a time (to compare).

With `--cpu-moe` the experts of an MoE model live in RAM and run on the CPU, the rest on the GPU; a layer is then about
4 times slower (RTX 5060 Ti 8 GB, Qwen3.5-35B-A3B: 0.93 ms against 0.24 ms). With layers from the coordinator the
video memory left after the rest of the slice (minus the headroom the coordinator keeps: 10%, at least 512 MB) goes to
the expert cache of llama.cpp (`--moe-cache-mib M` to set it): of the 256 experts of a layer 8 work per token and the
same ones are needed often, the cache keeps them on the GPU. `--gpu-experts N` instead keeps the experts of the last
N layers there. bench-batch, layers 0:24, 4 sessions: all experts on the CPU 74 tokens/s, 10 layers of experts on the
GPU 103, a cache of 3 GB 172. The status tells `moe_cache_mb` and `gpu_expert_layers`.

A forward with `FORWARD_CHECKPOINT` makes every node keep the state of the session after it (the recurrent layers;
4 per session, the oldest goes first); a later forward may start at that position again. `serve` keeps the chain of
the last request open (`--keep-chain-s 180`, a session at most 8 minutes) with checkpoints where the system message
ends and where the conversation ends before the answer: the next step of the agent reads only the answer and the tool
results, a new conversation only its first message. The coordinator takes the reports of a session as a running total.

Every 30 s the node prints `agent-node: status {...}` (the app shows it) and sends the same in the heartbeat
(`GET /v1/nodes` of the coordinator): sessions, `max_sessions`, forwards and those in shared passes, free video memory,
and on Windows the shared memory of the node (`vram_shared_mb`, as in Task Manager). When the weights or the context
do not fit, the Windows driver moves the rest to shared memory and the part runs several times slower: the node warns,
and with layers from the coordinator (`--model-url`) it asks at once for fewer (up to two times).

## Security

- **Access tokens.** A node started with `--key HEX` (or `AGENT_NODE_KEY`) accepts a session (`open`) and a previous
  node (`attach`) only with a token `<expiry>:<HMAC-SHA256(key, "agent-node/1|<session>|<expiry>")>`. The coordinator
  keeps the keys and issues the tokens (`agent-node token`); in tests the client issues them from `--keys`.
- **TLS 1.3** with `--tls` on all connections, client to node and node to node. Every node has its own key pair and
  self-signed certificate (`--identity node.pem`, created on first use). The client checks the certificate
  fingerprints given by `--pins`, and passes the fingerprint of the next node to every node.

```sh
agent-node keygen                                  # secret key of a node
agent-node identity --identity n0.pem              # prints the fingerprint
agent-node node -m model.gguf --layers 0:12 --tls --identity n0.pem --key KEY0
agent-node client -m model.gguf --chain A,B,C --tls --keys K0,K1,K2 --pins FP0,FP1,FP2
```

## Failover

The client watches all connections. A closed connection, an error of any node or no answer within `--timeout`
(3000 ms plus 20 ms per token of the batch) is a failure of the chain. The client then checks the nodes
(`hello`, in parallel), replaces the dead ones by `--spare` nodes that run the same layers, opens the session again
and reads the context once more in batches. The generation continues where it stopped; the text is the same.

```sh
agent-node client -m model.gguf --chain A,B,C --spare D --spare-keys KD --spare-pins FPD ...
```

Not done yet (see the transport spec of the agent repository): replay of the forward history by the previous node
instead of reading the whole context again (long contexts), int8 hidden states.

Supported architectures: those with layer range support (llama, qwen35, qwen35moe).

## Coordinator

The coordinator (`server/cmd/coordinator` in the agent repository) hands out chains, so the client needs no list
of nodes, keys or pins:

```sh
agent-node node -m model.gguf --layers 0:12 --tls --identity n0.pem --key HEX --coordinator http://coord:8080 \
    [--public-addr HOST:PORT] [--city NAME]
agent-node client -m model.gguf --coordinator http://coord:8080 -p "Hello" [--spares 1]
```

- The node measures its milliseconds per layer per token, then registers its address, certificate fingerprint,
  key, model key (`agent-node info`) and layer range. The coordinator connects to it with that pin and checks
  `hello` before it hands the node out. Heartbeats are signed: `Node <expiry>:<hex HMAC(key, "agent-coord/1|<id>|<expiry>")>`.
- The client asks `POST /v1/chains` with the model key; the reply has the session, the nodes and spares with
  their addresses, pins and session tokens.

## Tests

`tests/run.sh build/bin/agent-node Qwen3.5-0.8B-Q8_0.gguf` runs two nodes of a small model of the same architecture on
the CPU and checks: several clients at once (shared passes, the answers as the reference's), speculative decoding with
rejected drafts in two waves (rollback, sequences of ended sessions reused), and `serve` with a fake coordinator
(`tests/fake_coordinator.py`): the next step of an agent and a new conversation keep their start in the chain and in the
draft model, the answers do not change, the reports of a session are a running total. CI (`.github/workflows/agent-node.yml`)
runs it for every pull request to `agent/node`, plus a build with MSVC without CUDA.

