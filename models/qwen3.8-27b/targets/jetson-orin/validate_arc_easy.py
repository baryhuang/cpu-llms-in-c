#!/usr/bin/env python3
"""Run the repository's five-case ARC-Easy generation smoke profile."""

import argparse
import json
import re
import sys
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--url", default="http://127.0.0.1:8199/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    with open(args.profile, encoding="utf-8") as source:
        profile = json.load(source)

    system_text = profile["inference"]["system_text"]
    results = []
    for case in profile["cases"]:
        choices = "\n".join(
            f"{label}. {text}" for label, text in case["choices"].items()
        )
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": f'{case["question"]}\n{choices}'},
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        request = urllib.request.Request(
            args.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            decoded = json.load(response)
        message = decoded["choices"][0]["message"]
        text = message.get("content", "")
        matches = re.findall(r"(?<![A-Za-z])Answer: ([A-D])(?![A-Za-z])", text)
        parsed = matches[-1] if matches else None
        results.append(
            {
                "id": case["id"],
                "expected": case["answerKey"],
                "parsed": parsed,
                "correct": parsed == case["answerKey"],
                "text": text,
                "reasoning_content": message.get("reasoning_content", ""),
                "usage": decoded.get("usage"),
                "timings": decoded.get("timings"),
            }
        )
        print(
            f'{case["id"]}: parsed={parsed} expected={case["answerKey"]}',
            file=sys.stderr,
            flush=True,
        )

    output = {
        "profile": profile["id"],
        "max_tokens": args.max_tokens,
        "strict_score": sum(result["correct"] for result in results),
        "case_count": len(results),
        "cases": results,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
