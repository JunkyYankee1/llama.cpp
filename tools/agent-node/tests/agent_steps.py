# Steps of an agent against the local API of serve: a question with a tool, the same conversation with the tool result,
# a new conversation with the same system message. Prints one JSON line per answer (text and cached tokens).
import json, sys, urllib.error, urllib.request

URL = "http://127.0.0.1:%s/v1/chat/completions" % (sys.argv[1] if len(sys.argv) > 1 else "8090")
tools = [{"type": "function", "function": {"name": "calc", "description": "Evaluate an arithmetic expression and return the result.",
          "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a text file of the project.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
system = {"role": "system", "content": "You are a helpful assistant. " + "Use the tools when they help. " * 40}

def ask(messages):
    body = {"messages": messages, "tools": tools, "max_tokens": 48, "temperature": 0, "seed": 1, "think": False}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode(), flush=True)
        sys.exit(1)
    m = r["choices"][0]["message"]
    u = r["usage"]
    calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or []]
    print(json.dumps({"content": m.get("content"), "tool_calls": calls, "prompt": u["prompt_tokens"],
                      "cached": u.get("prompt_tokens_details", {}).get("cached_tokens", 0)}), flush=True)
    return m

conv = [system, {"role": "user", "content": "What is 1234 * 5678? Use the calc tool."}]
m = ask(conv)
conv.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls")})
for tc in m.get("tool_calls") or []:
    conv.append({"role": "tool", "tool_call_id": tc["id"], "content": "7006652"})
if not m.get("tool_calls"):
    conv.append({"role": "user", "content": "Now say the result again."})
ask(conv)
ask([system, {"role": "user", "content": "Say hello in French."}])
