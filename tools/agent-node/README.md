# agent-node

One part of a model in a chain of nodes, and the client that drives the chain.

```
client --tokens--> node A [layers 0..a) --fp16 hidden--> node B [a..b) --> ... --> node K [k..n) --top-k logits--> client
```

Every node runs a layer range of the model (`llama_model_params.layer_start/layer_end`) and loads only the
weights of its layers. Per token each node receives one frame and sends one frame: the hidden state of the token
in fp16 (`2 * n_embd` bytes plus a 27-byte header). The last node returns the top-k logits to the client.

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

Not done yet (see the transport spec of the agent repository): TLS and coordinator tokens, failover with replay
of the forward history, speculative decoding with state checkpoints, int8 hidden states.

Supported architectures: those with layer range support (llama, qwen35, qwen35moe).
