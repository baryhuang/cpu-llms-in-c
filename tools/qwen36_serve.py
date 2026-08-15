#!/usr/bin/env python3
"""OpenAI-compatible chat server for the resident Qwen3.6-27B runtime.

Wraps the resident C chat binary (machine protocol mode) behind
POST /v1/chat/completions with SSE streaming and GET /v1/models, so any
OpenAI-compatible client - Chatbox, Cherry Studio, Open WebUI, Raycast,
Continue, Cline - can talk to the local model. Standard library only;
the shim renders the official chat template (thinking optional) and owns the HTTP
protocol, while all model execution stays in the C/Metal runtime.

Usage:
  tools/qwen36_serve.py                     # serve on 127.0.0.1:8199
  tools/qwen36_serve.py --port 9000
  curl http://127.0.0.1:8199/v1/chat/completions -d '{
    "model": "qwen3.6-27b", "stream": true,
    "messages": [{"role": "user", "content": "hello"}]}'

Then point a client at base URL http://127.0.0.1:8199/v1 with any API
key. Requests are served one at a time; the model loads and wires once
at startup, so every request runs at ready-state latency.
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "qwen3.6-27b"


REASONING_EFFORT_TEXT = {
    "xhigh": ("Reasoning effort is set to xhigh. Please think carefully "
              "through the task, validate key assumptions, consider "
              "plausible alternatives, and prioritize correctness, "
              "consistency, and clarity in the final answer."),
    "low": ("Reasoning effort is set to low. Keep your thinking brief "
            "and focused, moving directly to the conclusion without "
            "unnecessary elaboration."),
    "medium": None,
}


def message_text(message):
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content
                          if isinstance(part, dict))
    return content


def render_template(messages, thinking=False, effort="xhigh"):
    """The official Qwen3.6/3.8 template. Default is the pinned
    enable_thinking=false rendering; thinking mode opens the reply at
    '<think>\n' and injects the Qwen3.8 reasoning-effort instructions
    (verbatim template strings) at the top of the system turn, creating
    one when the effort is not medium and no system message exists.

    Prior assistant turns replay the think block the model actually saw
    (empty in no-think mode, or the turn's reasoning_content) — the
    Qwen3.8 preserve_thinking semantics. This also makes each follow-up
    request an exact token extension of the resident conversation state,
    so the engine prefills only the new turn instead of the whole
    history."""
    parts = []
    remaining = list(messages)
    instructions = REASONING_EFFORT_TEXT.get(effort) if thinking else None
    if remaining and remaining[0].get("role") == "system":
        content = message_text(remaining[0]).strip()
        remaining = remaining[1:]
        head = (instructions + "\n\n" if instructions else "") + content
        if head:
            parts.append(f"<|im_start|>system\n{head}<|im_end|>\n")
    elif instructions:
        parts.append(f"<|im_start|>system\n{instructions}<|im_end|>\n")
    for message in remaining:
        role = message.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = message_text(message)
        if role == "assistant":
            reasoning = message.get("reasoning_content") or ""
            parts.append(f"<|im_start|>assistant\n<think>\n{reasoning}\n"
                         f"</think>\n\n{content}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if thinking:
        parts.append("<|im_start|>assistant\n<think>\n")
    else:
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


class ThinkSplitter:
    """Split a streamed reply into reasoning and answer. In thinking
    mode the generation begins inside the think block; everything before
    '</think>' is reasoning_content, everything after (minus the
    separating blank line) is content. Holds back a possible partial tag
    at a delta boundary."""

    def __init__(self, active):
        self.active = active
        self.buffer = ""
        self.done = not active

    def feed(self, text):
        if self.done:
            return ("", text) if self.active else (None, text)
        self.buffer += text
        marker = self.buffer.find("</think>")
        if marker >= 0:
            reasoning = self.buffer[:marker]
            rest = self.buffer[marker + len("</think>"):]
            rest = rest.lstrip("\n")
            self.buffer = ""
            self.done = True
            return (reasoning, rest)
        # hold back a suffix that could start the closing tag
        keep = 0
        for probe in range(min(len("</think>") - 1, len(self.buffer)), 0, -1):
            if "</think>".startswith(self.buffer[-probe:]):
                keep = probe
                break
        emit = self.buffer[:len(self.buffer) - keep]
        self.buffer = self.buffer[len(self.buffer) - keep:]
        return (emit, "")

    def flush(self):
        rest = self.buffer
        self.buffer = ""
        return rest


class Engine:
    """One resident chat process, one request at a time."""

    def __init__(self, arguments):
        self.arguments = arguments
        self.lock = threading.Lock()
        self.process = None
        self.ready = {}

    def start(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        binary = os.path.join(repo, "build", "qwen36-m3-chat")
        model = self.arguments.model_dir or os.path.join(
            repo, "tmp", "qwen36-27b-runtime")
        command = [
            binary, model,
            self.arguments.metallib or os.path.join(
                repo, "build", "qwen36-m3-q4.metallib"),
            self.arguments.tokenizer or os.path.join(
                model, "tokenizer.q36tok"),
            str(self.arguments.context), str(self.arguments.max_tokens),
            str(self.arguments.temperature), str(self.arguments.top_k),
            str(self.arguments.seed),
        ]
        environment = dict(os.environ, QWEN36_MACHINE="1")
        print(f"loading model (one-time weight wiring)...", flush=True)
        self.process = subprocess.Popen(
            command, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
            bufsize=1)
        line = self.process.stdout.readline()
        if not line.startswith("R "):
            raise RuntimeError(f"engine did not become ready: {line!r}")
        self.ready = json.loads(line[2:])
        print(f"model ready: context {self.ready.get('context')}, "
              f"max reply {self.ready.get('max_new')} tokens", flush=True)

    def generate(self, rendered, on_delta, sampling=None):
        """Run one request; call on_delta(text) per chunk; return stats."""
        with self.lock:
            if self.process.poll() is not None:
                self.start()
            if sampling:
                request = dict(sampling)
                request["prompt"] = rendered
                self.process.stdin.write(json.dumps(request) + "\n")
            else:
                self.process.stdin.write(json.dumps(rendered) + "\n")
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("engine exited mid-request")
                if line.startswith("D "):
                    on_delta(json.loads(line[2:]))
                elif line.startswith("E "):
                    return json.loads(line[2:])
                elif line.startswith("X "):
                    raise RuntimeError(json.loads(line[2:]))


ENGINE = None
THINKING_DEFAULT = False
EFFORT_DEFAULT = "xhigh"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *values):
        print(f"{self.address_string()} {format % values}", flush=True)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self.send_json({"object": "list", "data": [{
                "id": MODEL_ID, "object": "model",
                "created": int(time.time()), "owned_by": "local"}]})
        elif self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length))
            messages = request["messages"]
        except (ValueError, KeyError):
            self.send_json({"error": {
                "message": "body must be JSON with a messages array",
                "type": "invalid_request_error"}}, 400)
            return
        # Thinking mode: the server default (--thinking /
        # QWEN36_THINKING) can be overridden per request with the
        # OpenAI-style reasoning_effort field ("none" disables thinking,
        # low/medium/xhigh select the Qwen3.8 effort levels; "high" maps
        # to xhigh).
        thinking = THINKING_DEFAULT
        effort = EFFORT_DEFAULT
        effort_field = request.get("reasoning_effort")
        if isinstance(effort_field, str):
            level = effort_field.lower()
            if level == "none":
                thinking = False
            elif level in ("low", "medium", "xhigh", "high"):
                thinking = True
                effort = "xhigh" if level == "high" else level
        rendered = render_template(messages, thinking, effort)
        # Per-request sampling passes straight through to the engine.
        # Absent fields keep the engine defaults (greedy, which also
        # enables lossless speculative decoding); a request that sets
        # temperature > 0 gets true sampled decoding with top_k, top_p,
        # min_p and presence_penalty honored.
        sampling = {}
        for source, target in (("temperature", "temperature"),
                               ("top_k", "top_k"),
                               ("top_p", "top_p"),
                               ("min_p", "min_p"),
                               ("presence_penalty", "presence_penalty"),
                               ("max_tokens", "max_new"),
                               ("max_completion_tokens", "max_new")):
            value = request.get(source)
            if isinstance(value, (int, float)) and not isinstance(
                    value, bool):
                sampling[target] = value
        identifier = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        if request.get("stream"):
            include_usage = bool(
                (request.get("stream_options") or {}).get("include_usage"))
            self.stream_completion(rendered, identifier, created,
                                   include_usage, sampling, thinking)
        else:
            self.plain_completion(rendered, identifier, created, sampling,
                                  thinking)

    def plain_completion(self, rendered, identifier, created,
                         sampling=None, thinking=False):
        chunks = []
        try:
            stats = ENGINE.generate(rendered, chunks.append, sampling)
        except RuntimeError as failure:
            self.send_json({"error": {"message": str(failure),
                                      "type": "server_error"}}, 500)
            return
        text = "".join(chunks)
        message = {"role": "assistant", "content": text}
        if thinking:
            marker = text.find("</think>")
            if marker >= 0:
                message["reasoning_content"] = text[:marker]
                message["content"] = text[marker + len("</think>"):] \
                    .lstrip("\n")
            else:
                message["reasoning_content"] = text
                message["content"] = ""
        self.send_json({
            "id": identifier, "object": "chat.completion",
            "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": message,
                "finish_reason": stats.get("stop", "stop")}],
            "usage": {
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "completion_tokens": stats.get("tokens", 0),
                "total_tokens": stats.get("prompt_tokens", 0) +
                                stats.get("tokens", 0)},
        })

    def stream_completion(self, rendered, identifier, created,
                          include_usage, sampling=None, thinking=False):
        # The SSE body carries no Content-Length, so it must use chunked
        # transfer encoding: without the 0-length terminator a keep-alive
        # client never sees the body end and its UI stays in the waiting
        # state after the reply.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def write_chunk(data):
            self.wfile.write(f"{len(data):x}\r\n".encode() + data +
                             b"\r\n")
            self.wfile.flush()

        def event(payload):
            write_chunk(f"data: {json.dumps(payload)}\n\n".encode())

        def chunk(delta, finish=None):
            return {"id": identifier, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}]}

        try:
            event(chunk({"role": "assistant", "content": ""}))
            splitter = ThinkSplitter(thinking)

            def deliver(text):
                reasoning, content = splitter.feed(text)
                if reasoning:
                    event(chunk({"reasoning_content": reasoning}))
                if content:
                    event(chunk({"content": content}))

            stats = ENGINE.generate(rendered, deliver, sampling)
            tail = splitter.flush()
            if tail:
                if splitter.done:
                    event(chunk({"content": tail}))
                else:
                    event(chunk({"reasoning_content": tail}))
            event(chunk({}, stats.get("stop", "stop")))
            if include_usage:
                prompt_tokens = stats.get("prompt_tokens", 0)
                completion_tokens = stats.get("tokens", 0)
                event({"id": identifier,
                       "object": "chat.completion.chunk",
                       "created": created, "model": MODEL_ID,
                       "choices": [],
                       "usage": {
                           "prompt_tokens": prompt_tokens,
                           "completion_tokens": completion_tokens,
                           "total_tokens": prompt_tokens +
                                           completion_tokens}})
            write_chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (RuntimeError, BrokenPipeError,
                ConnectionResetError) as failure:
            print(f"stream aborted: {failure}", flush=True)
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible server for the resident "
                    "Qwen3.6-27B C/Metal runtime.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--model-dir")
    parser.add_argument("--metallib")
    parser.add_argument("--tokenizer")
    parser.add_argument("--context", type=int,
                        default=int(os.environ.get("QWEN36_CONTEXT",
                                                   4096)))
    parser.add_argument("--max-tokens", type=int,
                        default=int(os.environ.get("QWEN36_MAX_TOKENS",
                                                   3072)))
    parser.add_argument("--thinking", action="store_true",
                        default=os.environ.get("QWEN36_THINKING",
                                               "") not in ("", "0"))
    parser.add_argument("--reasoning-effort",
                        default=os.environ.get("QWEN36_REASONING_EFFORT",
                                               "xhigh"),
                        choices=["low", "medium", "xhigh"])
    parser.add_argument("--temperature", type=float,
                        default=float(os.environ.get("QWEN36_TEMPERATURE",
                                                     0)))
    parser.add_argument("--top-k", type=int,
                        default=int(os.environ.get("QWEN36_TOP_K", 1)))
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("QWEN36_SEED", 42)))
    arguments = parser.parse_args()

    global ENGINE, THINKING_DEFAULT, EFFORT_DEFAULT
    THINKING_DEFAULT = bool(arguments.thinking)
    EFFORT_DEFAULT = arguments.reasoning_effort
    ENGINE = Engine(arguments)
    ENGINE.start()

    # Take the resident engine down with the server: without this a
    # killed server orphans a child process holding the wired model.
    def shutdown(*_):
        process = ENGINE.process if ENGINE else None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    atexit.register(shutdown)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    server = ThreadingHTTPServer((arguments.host, arguments.port),
                                 Handler)
    print(f"serving OpenAI-compatible API at "
          f"http://{arguments.host}:{arguments.port}/v1 "
          f"(model id '{MODEL_ID}')", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
