"""OpenAI chat-completions shim in front of `needle --serve`.

Needle's server holds one conversation, fixes tools and system facts at startup, and
takes `POST /complete {"input": ...}` where input is the user query on the first turn
and a tool-result JSON on later turns. It never generates prose: its documented
contract is that on a `respond` turn "the answer is the tool results". The shim maps:

  first request of a conversation (no assistant message yet) -> /reset, then user text
  trailing role:tool messages -> one /complete with their contents (JSON array if >1)
  type=call with function_calls -> message.tool_calls
  anything else -> message.content = model reasoning, plus on `respond` turns the tool
                   results it was just handed (Needle's own definition of its answer)
"""

import itertools
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

NEEDLE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
ids = itertools.count(1)


def needle(path, body):
    req = urllib.request.Request(NEEDLE + path, json.dumps(body, separators=(",", ":")).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        text = r.read().decode()
    return json.loads(text) if text.strip() else {}


def turn(messages):
    if not any(m["role"] == "assistant" for m in messages):
        needle("/reset", {})
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        return needle("/complete", {"input": user}), None
    results = []
    for m in reversed(messages):
        if m["role"] != "tool":
            break
        results.insert(0, m["content"])
    if not results:  # a new user turn in an existing conversation
        return needle("/complete", {"input": messages[-1]["content"]}), None
    payload = results[0] if len(results) == 1 else "[" + ",".join(results) + "]"
    return needle("/complete", {"input": payload}), payload


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "needle3", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out, fed = turn(body["messages"])
        sys.stderr.write(json.dumps(out) + "\n")
        message = {"role": "assistant", "content": ""}
        calls = out.get("function_calls") or []
        if out.get("type") == "call" and calls:
            message["tool_calls"] = [
                {"id": f"call_{next(ids)}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
                for c in calls
            ]
        else:
            text = out.get("reasoning") or out.get("error") or ""
            if out.get("type") == "respond" and fed:
                text = f"{text}\n\n{fed}"
            message["content"] = text
        self._send({
            "id": f"chatcmpl-{next(ids)}", "object": "chat.completion", "model": "needle3",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
            "timings": {"predicted_per_second": out.get("decode_tps") or None,
                        "prompt_per_second": out.get("prefill_tps"),
                        "peak_ram_mb": out.get("peak_ram_mb")},
        })

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
