# A coordinator for the tests of serve: one account, the chain of the nodes given on the command line, the reports.
#   python3 fake_coordinator.py PORT ADDR=PIN [ADDR=PIN ...]
# The nodes run without --key (open): their tokens are empty. Reports must be a running total per session
# (serve keeps one session for several requests); a report that goes down is an error, printed and counted.
import json, secrets, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(sys.argv[1])
nodes = [{"addr": a, "pin": p, "token": ""} for a, p in (x.split("=", 1) for x in sys.argv[2:])]
reports = {}  # session -> (prompt, gen)

class H(BaseHTTPRequestHandler):
    def reply(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/v1/accounts":
            return self.reply(200, {"account_id": "test", "token": "t", "balance": 1000})
        if self.path == "/v1/chains":
            return self.reply(200, {"session": secrets.token_hex(8), "nodes": nodes, "spares": [], "stages_without_spare": len(nodes)})
        if self.path.startswith("/v1/sessions/") and self.path.endswith("/report"):
            sid = self.path.split("/")[3]
            p, g = body["prompt_tokens"], body["gen_tokens"]
            old = reports.get(sid, (0, 0))
            if p < old[0] or g < old[1]:
                print(f"fake-coordinator: ERROR report of {sid} went down: {old} -> {(p, g)}", flush=True)
            reports[sid] = (p, g)
            print(f"fake-coordinator: report {sid} prompt {p} gen {g}", flush=True)
            return self.reply(200, {"charged": 0, "balance": 1000})
        self.reply(404, {"error": {"message": "unknown path " + self.path}})

    def log_message(self, *a):
        pass

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
