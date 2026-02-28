"""
analyze.py — entry point for dialogue analysis.

Usage:
    python analyze.py --input output/final/ --output output/final/analysis.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env", override=True)

from config import BACKENDS, MODELS, PROMPTS
from core.llm_client import LLMClient, safe_json_parse
from core.models import (
    AnalysisResult,
)


def analyze_dialogue(client: LLMClient, dialogue: dict) -> AnalysisResult:
    """Run QA analysis on a single dialogue dict."""
    messages = dialogue.get("messages", [])
    lines = []
    for m in messages:
        role = "Client" if m["role"] == "client" else "Agent"
        lines.append(f"[Turn {m['turn']}] {role}: {m['content']}")
    dialogue_text = "\n".join(lines)

    raw = client.complete(
        system=PROMPTS["analysis_system"],
        user=PROMPTS["analysis_user"].format(dialogue_text=dialogue_text),
        model=MODELS["parser"],
        max_tokens=512,
        backend=BACKENDS["parser"],
    )

    data = safe_json_parse(raw)

    # Coerce agent_mistakes to list[str]
    mistakes = data.get("agent_mistakes", [])
    if isinstance(mistakes, str):
        mistakes = [mistakes] if mistakes else []

    return AnalysisResult(
        dialogue_id=dialogue.get("id", "unknown"),
        intent=data.get("intent", "other"),
        satisfaction=data.get("satisfaction", "neutral"),
        quality_score=int(data.get("quality_score", 3)),
        agent_mistakes=mistakes,
        hidden_dissatisfaction=bool(data.get("hidden_dissatisfaction", False)),
        resolution=data.get("resolution", "unresolved"),
        summary=data.get("summary", ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated dialogues")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with dialogue JSON files",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output analysis JSON file"
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer combined file if exists, otherwise glob individuals
    combined = in_dir / "all_dialogues.json"
    if combined.exists():
        with open(combined, encoding="utf-8") as f:
            dialogues = json.load(f)
        files = [combined] * len(dialogues)
    else:
        json_files = sorted(f for f in in_dir.glob("dialogue_*.json"))
        dialogues = []
        files = []
        for jf in json_files:
            try:
                with open(jf, encoding="utf-8") as f:
                    dialogues.append(json.load(f))
                files.append(jf)
            except Exception:
                pass

    if not dialogues:
        print(f"No dialogues found in {in_dir}")
        sys.exit(1)

    client = LLMClient()
    results = []
    errors = []
    t_start = time.time()

    print(f"Analyzing {len(dialogues)} dialogue(s) from {in_dir}")
    print("─" * 50)

    for i, dialogue in enumerate(dialogues):
        did = dialogue.get("id", f"idx-{i}")
        print(f"[{i + 1:03d}/{len(dialogues)}] {did[:20]}...", end=" ", flush=True)
        try:
            ar = analyze_dialogue(client, dialogue)
            results.append(ar.model_dump())
            print(
                f"✓  intent={ar.intent.value}  score={ar.quality_score}  "
                f"sat={ar.satisfaction.value}"
            )
        except Exception as e:
            print(f"✗  {e}")
            errors.append({"dialogue_id": did, "error": str(e)})
            results.append(
                {
                    "dialogue_id": did,
                    "intent": "other",
                    "satisfaction": "neutral",
                    "quality_score": 0,
                    "agent_mistakes": [],
                    "hidden_dissatisfaction": False,
                    "resolution": "unresolved",
                    "summary": f"Analysis failed: {str(e)[:80]}",
                }
            )

    elapsed = round(time.time() - t_start, 1)

    # ── compute stats ──────────────────────────────────────────
    valid = [r for r in results if r["quality_score"] > 0]
    avg_q = (
        round(sum(r["quality_score"] for r in valid) / len(valid), 2) if valid else 0
    )

    intent_dist: dict[str, int] = {}
    sat_dist: dict[str, int] = {}
    res_dist: dict[str, int] = {}
    for r in results:
        intent_dist[r["intent"]] = intent_dist.get(r["intent"], 0) + 1
        sat_dist[r["satisfaction"]] = sat_dist.get(r["satisfaction"], 0) + 1
        res_dist[r["resolution"]] = res_dist.get(r["resolution"], 0) + 1

    stats = {
        "total_analyzed": len(results),
        "errors": len(errors),
        "avg_quality_score": avg_q,
        "hidden_dissatisfaction_count": sum(
            1 for r in results if r["hidden_dissatisfaction"]
        ),
        "dialogues_with_mistakes": sum(1 for r in results if r["agent_mistakes"]),
        "intent_distribution": intent_dist,
        "satisfaction_distribution": sat_dist,
        "resolution_distribution": res_dist,
        "elapsed_s": elapsed,
    }

    output = {"stats": stats, "results": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("─" * 50)
    print(f"Done: {len(results)} analyzed, {len(errors)} errors — {elapsed}s")
    print(f"Avg quality:      {avg_q}/5")
    print(f"Hidden dissatisf: {stats['hidden_dissatisfaction_count']}")
    print(f"With mistakes:    {stats['dialogues_with_mistakes']}")
    print(f"Output: {out_path}")


# Exported for self_eval.py
def analyze_dialogue_exported(client: LLMClient, dialogue: dict) -> AnalysisResult:
    return analyze_dialogue(client, dialogue)


if __name__ == "__main__":
    main()
