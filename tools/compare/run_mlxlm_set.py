"""Five-workload set through bare mlx-lm: resident model, one warmup,
then the five requests with per-request wall and reply token counts."""
import json
import sys
import time

sys.path.insert(0, "/Users/buryhuang/git/cpullama/tools/compare")
from five_workloads import WORKLOADS, WARMUP, render

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

model_path, out_path = sys.argv[1], sys.argv[2]
model, tokenizer = load(model_path)
sampler = make_sampler(temp=0)


def run(user_text, max_tokens):
    prompt = render(user_text)
    start = time.time()
    count = 0
    last = start
    for response in stream_generate(model, tokenizer, prompt=prompt,
                                    max_tokens=max_tokens, sampler=sampler):
        count += 1
        last = time.time()
    return count, last - start


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
summary = {"stack": "mlx-lm 0.31.3 / mlx 0.32.0", "model": model_path,
           "cases": rows, "total_reply_tokens": total_tokens,
           "total_wall_s": round(total_wall, 1),
           "e2e_tok_s": round(total_tokens / total_wall, 2)}
json.dump(summary, open(out_path, "w"), indent=1)
print("AGGREGATE", summary["e2e_tok_s"], "tok/s")
