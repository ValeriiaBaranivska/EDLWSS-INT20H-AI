"""
Generate balanced batch of dialogues with diverse topics/outcomes.
Usage: python generate_batch.py [count] [base_seed]
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from core.models import DiceRollParams
from core.pipeline import Pipeline
from pipeline import stage0_dice

# Balanced scenarios for diversity (20 unique combinations)
SCENARIOS = [
    # Topic, Outcome, Style
    # payment_issue (4 variants)
    ("payment_issue", "resolved_quick", "formal"),
    ("payment_issue", "unresolved_passive", "casual"),
    ("payment_issue", "conflict", "aggressive"),
    ("payment_issue", "resolved_neutral", "passive_aggressive"),
    # technical_error (4 variants)
    ("technical_error", "resolved_neutral", "casual"),
    ("technical_error", "conflict", "aggressive"),
    ("technical_error", "unresolved_ragequit", "formal"),
    ("technical_error", "resolved_quick", "casual"),
    # account_access (4 variants)
    ("account_access", "resolved_quick", "formal"),
    ("account_access", "unresolved_ragequit", "aggressive"),
    ("account_access", "resolved_neutral", "casual"),
    ("account_access", "unresolved_passive", "passive_aggressive"),
    # billing_question (4 variants)
    ("billing_question", "info_only", "casual"),
    ("billing_question", "resolved_neutral", "formal"),
    ("billing_question", "conflict", "aggressive"),
    ("billing_question", "resolved_quick", "passive_aggressive"),
    # refund_request (4 variants)
    ("refund_request", "unresolved_passive", "passive_aggressive"),
    ("refund_request", "resolved_quick", "casual"),
    ("refund_request", "conflict", "aggressive"),
    ("refund_request", "resolved_neutral", "formal"),
]


def generate_balanced(count: int, base_seed: int):
    """Generate dialogues with balanced scenarios."""
    pipe = Pipeline()
    dialogues = []

    print(f"Generating {count} balanced dialogues...")
    print(f"Base seed: {base_seed}\n")

    for i in range(count):
        seed = base_seed + i * 100
        scenario = SCENARIOS[i % len(SCENARIOS)]
        topic, outcome, style = scenario

        # Override dice params for balanced distribution
        params = stage0_dice.run(seed)
        params = params.model_copy(
            update={
                "topic": topic,
                "outcome": outcome,
                "style": style,
            }
        )

        print(f"=== Generating {i + 1}/{count} (seed={seed}) ===")
        print(f"  Topic: {topic}, Outcome: {outcome}, Style: {style}")

        try:
            # Run pipeline with overridden params
            result = _run_with_params(pipe, params, seed)
            dialogues.append(result)
            print(f"  ✓ {len(result.messages)} messages")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    return dialogues


def _run_with_params(pipe: Pipeline, params: DiceRollParams, seed: int):
    """Run pipeline with custom params (bypass dice)."""
    import time
    import uuid

    from config import MODELS
    from core.models import DialogueOutput
    from pipeline import stage1_characters, stage5_simple_style, stage6_dirt

    t0 = time.time()
    stages_run = ["0"]
    stages_skipped = ["2a", "2b", "3", "4", "4B"]

    # Stage 1: characters
    characters = stage1_characters.run(pipe.client, params)
    stages_run.append("1")

    # Stage 5: simple generation with qwen
    messages = stage5_simple_style.run(pipe.client, params, characters)
    stages_run.append("5")

    if len(messages) < 2:
        raise ValueError("Too few messages generated")

    # Strip bot phrases
    from core.pipeline import strip_bot_phrases

    messages = strip_bot_phrases(messages)

    # Stage 6: dirt
    messages, dirt_applied = stage6_dirt.DirtLayer(
        slang_bank_path="corpus/slang_bank.json",
        filler_bank_path="corpus/filler_bank.json",
        seed=seed,
    ).apply(messages, params)
    stages_run.append("6")

    return DialogueOutput(
        id=str(uuid.uuid4()),
        seed=seed,
        params=params,
        characters=characters,
        messages=messages,
        meta={
            "total_time_s": round(time.time() - t0, 2),
            "model_used": MODELS["styler"],
            "stages_run": stages_run,
            "stages_skipped": stages_skipped,
            "dirt_applied": dirt_applied,
        },
    )


def save_outputs(dialogues, output_dir: str = "output/batch"):
    """Save dialogues as txt + json."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # JSON with full data
    json_file = out_path / "dialogues_batch.json"
    json_data = {
        "generated": datetime.now().isoformat(),
        "count": len(dialogues),
        "dialogues": [d.model_dump() for d in dialogues],
    }
    json_file.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nJSON saved: {json_file}")

    # TXT with dialogues only
    txt_file = out_path / "dialogues_batch.txt"
    lines = [
        f"Generated: {datetime.now().isoformat()}",
        f"Count: {len(dialogues)}",
        "=" * 60,
        "",
    ]
    for d in dialogues:
        lines.append(f"--- Dialogue (seed={d.seed}) ---")
        lines.append(f"Topic: {d.params.topic}")
        lines.append(f"Outcome: {d.params.outcome}")
        lines.append(f"Style: {d.params.style}")
        lines.append("")
        for m in d.messages:
            role = "Client" if m.role.value == "client" else "Agent"
            lines.append(f"{role}: {m.content}")
        lines.append("")

    txt_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"TXT saved: {txt_file}")

    return json_file, txt_file


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    base_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 100000

    dialogues = generate_balanced(count, base_seed)

    if dialogues:
        save_outputs(dialogues)
        print(f"\n✓ Generated {len(dialogues)} dialogues")
    else:
        print("No dialogues generated")


if __name__ == "__main__":
    main()
