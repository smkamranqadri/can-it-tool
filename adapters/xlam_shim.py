"""OpenAI tool-calling adapter for xLAM-2 on llama-server.

llama.cpp cannot parse xLAM's tool-call output (a bare JSON array in content), and the
model's own template drops its format instruction whenever a system prompt is given.
This shim does what xLAM's template does on its default path: it removes `tools` from
the request, appends xLAM's format instruction and the tool schemas to the system
message, forwards to llama-server, and turns a JSON-array reply into `tool_calls`.
Everything else - the system prompt, tool results, sampling - passes through unchanged.
"""

import itertools
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

UPSTREAM = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
ids = itertools.count(1)

FORMAT = """You have access to a set of tools. When using tools, make calls in a single JSON array:

[{"name": "tool_call_name", "arguments": {"arg1": "value1", "arg2": "value2"}}, ... (additional parallel tool calls as needed)]

If no tool is suitable, state that explicitly. If the user's input lacks required parameters, ask for clarification. Do not interpret or respond until tool results are returned. Once they are available, process them or make additional calls if needed. For tasks that don't require tools, such as casual conversation or general advice, respond directly in plain text. The available tools are:"""


def parse_calls(text):
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").removeprefix("json").strip()
    if not s.startswith("["):
        return None
    try:
        calls = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(calls, list) or not all(isinstance(c, dict) and "name" in c for c in calls):
        return None
    return calls


def rewrite(body):
    tools = body.pop("tools", None)
    body.pop("tool_choice", None)
    messages = []
    for m in body["messages"]:
        m = dict(m)
        if m["role"] == "assistant" and m.get("tool_calls"):
            # xLAM sees its own earlier calls in its own format
            m["content"] = json.dumps([
                {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"] or "{}")}
                for c in m.pop("tool_calls")])
        messages.append(m)
    if tools:
        schemas = "\n\n".join(json.dumps(t["function"], indent=4) for t in tools)
        block = f"{FORMAT}\n\n{schemas}"
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + block
        else:
            messages.insert(0, {"role": "system", "content": block})
    body["messages"] = messages
    return body


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        with urllib.request.urlopen(UPSTREAM + self.path) as r:
            self._send(r.status, r.read())

    def do_POST(self):
        body = rewrite(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        req = urllib.request.Request(UPSTREAM + self.path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            return self._send(e.code, e.read())
        msg = out["choices"][0]["message"]
        calls = parse_calls(msg.get("content") or "")
        if calls is not None:
            msg["content"] = ""
            msg["tool_calls"] = [
                {"id": f"call_{next(ids)}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
                for c in calls]
            out["choices"][0]["finish_reason"] = "tool_calls"
        self._send(200, json.dumps(out).encode())

    def _send(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
