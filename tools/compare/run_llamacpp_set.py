"""Five-workload set against a resident llama-server (started by the
caller): one warmup, then the five requests via /completion."""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/Users/buryhuang/git/cpullama/tools/compare")
from five_workloads import WORKLOADS, WARMUP, render

port, out_path = sys.argv[1], sys.argv[2]


def run(user_text, max_tokens):
    body = json.dumps({"prompt": render(user_text),
                       "n_predict": max_tokens, "temperature": 0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", body,
        {"Content-Type": "application/json"})
    start = time.time()
    data = json.loads(urllib.request.urlopen(req, timeout=1200).read())
    wall = time.time() - start
    tokens = data.get("timings", {}).get("predicted_n", 0)
    return tokens, wall


run(WARMUP, 16)
rows = []
for name, text in WORKLOADS:
    tokens, wall = run(text, 1800)
    rows.append({"case": name, "reply_tokens": tokens,
                 "wall_s": round(wall, 3),
                 "e2e_tok_s": round(tokens / wall, 2)})
    print(rows[-1], flush=True)
total_tokens = sum(r["reply_tokens"] for r in rows)
total_wall = sum(r["wall_s"] for r in rows)
summary = {"stack": "llama.cpp b10360 llama-server", "port": port,
           "cases": rows, "total_reply_tokens": total_tokens,
           "total_wall_s": round(total_wall, 1),
           "e2e_tok_s": round(total_tokens / total_wall, 2)}
json.dump(summary, open(out_path, "w"), indent=1)
print("AGGREGATE", summary["e2e_tok_s"], "tok/s")
