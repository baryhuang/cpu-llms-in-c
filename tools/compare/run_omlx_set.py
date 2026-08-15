"""Five-workload set through the oMLX BatchedEngine: resident engine,
enable_thinking=False to match the no-think comparison, one warmup, then
the five requests."""
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from five_workloads import WORKLOADS, WARMUP

from omlx.engine import BatchedEngine

model_path, out_path = sys.argv[1], sys.argv[2]


async def main():
    engine = BatchedEngine(model_path, enable_thinking=False,
                           stream_interval=1)
    await engine.start()

    async def run(user_text, max_tokens):
        messages = [{"role": "user", "content": user_text}]
        start = time.time()
        events = 0
        last = start
        final = None
        async for output in engine.stream_chat(messages,
                                               max_tokens=max_tokens,
                                               temperature=0.0):
            events += 1
            last = time.time()
            final = output
        tokens = events
        for attr in ("completion_tokens", "generated_tokens", "num_tokens"):
            value = getattr(final, attr, None)
            if isinstance(value, int) and value > 0:
                tokens = value
                break
        usage = getattr(final, "usage", None)
        if isinstance(usage, dict) and usage.get("completion_tokens"):
            tokens = usage["completion_tokens"]
        return tokens, last - start

    await run(WARMUP, 16)
    rows = []
    for name, text in WORKLOADS:
        tokens, wall = run_result = await run(text, 1800)
        rows.append({"case": name, "reply_tokens": tokens,
                     "wall_s": round(wall, 3),
                     "e2e_tok_s": round(tokens / wall, 2)})
        print(rows[-1], flush=True)
    await engine.stop()
    total_tokens = sum(r["reply_tokens"] for r in rows)
    total_wall = sum(r["wall_s"] for r in rows)
    summary = {"stack": "oMLX 0.5.7 BatchedEngine", "model": model_path,
               "cases": rows, "total_reply_tokens": total_tokens,
               "total_wall_s": round(total_wall, 1),
               "e2e_tok_s": round(total_tokens / total_wall, 2)}
    json.dump(summary, open(out_path, "w"), indent=1)
    print("AGGREGATE", summary["e2e_tok_s"], "tok/s")


asyncio.run(main())
