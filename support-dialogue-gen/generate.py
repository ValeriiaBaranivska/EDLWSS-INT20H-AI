"""
generate.py — entry point for dialogue generation.

Usage:
    python generate.py --count 50 --output output/final/
    python generate.py --count 1 --seed 12345 --output output/debug/
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env", override=True)

from core.pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate customer support dialogues")
    parser.add_argument(
        "--count", type=int, default=20, help="Number of dialogues to generate"
    )
    parser.add_argument(
        "--output", type=str, default="output/batch/", help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Fixed seed (single dialogue)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline()
    count = 1 if args.seed is not None else args.count

    all_dialogues = []
    errors = []
    t_start = time.time()

    print(f"Generating {count} dialogue(s) → {out_dir}")
    print("─" * 50)

    for i in range(count):
        seed = args.seed if args.seed is not None else None
        print(f"[{i + 1:03d}/{count}] generating...", end=" ", flush=True)
        try:
            result = pipe.run(seed=seed)
            d = result.model_dump()

            # individual file
            fname = out_dir / f"dialogue_{i + 1:03d}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)

            all_dialogues.append(d)
            turns = len([m for m in d["messages"] if m["role"] == "client"])
            print(
                f"✓  seed={d['seed']}  turns={turns}  "
                f"{d['params']['complexity']}/{d['params']['sector']}"
            )
        except Exception as e:
            print(f"✗  ERROR: {e}")
            errors.append({"index": i + 1, "error": str(e)})

    # write combined file
    all_path = out_dir / "all_dialogues.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_dialogues, f, indent=2, ensure_ascii=False)

    elapsed = round(time.time() - t_start, 1)
    print("─" * 50)
    print(f"Done: {len(all_dialogues)} generated, {len(errors)} errors — {elapsed}s")
    print(f"Output: {all_path}")
    if errors:
        print(f"Errors: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()
