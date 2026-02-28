"""
Generate sample dialogues with specific constraints.
Outputs:
  - dialogues_sample.txt  — plain text with just the dialogues
  - dialogues_sample.json — full JSON with all metadata and logs
"""

import json
import random
import sys
from datetime import datetime

from core.pipeline import Pipeline

# Force non-conflict outcomes
NON_CONFLICT_OUTCOMES = [
    "resolved_quick",
    "resolved_neutral",
    "info_only",
]


def generate_non_conflict_seed(base_seed: int) -> int:
    """
    Find a seed that produces a non-conflict outcome.
    We test seeds starting from base_seed until we find one.
    """
    from pipeline.stage0_dice import run as dice_roll

    for offset in range(1000):
        test_seed = base_seed + offset
        params = dice_roll(test_seed)
        if params.outcome in NON_CONFLICT_OUTCOMES:
            return test_seed
    return base_seed  # fallback


def format_dialogue_text(dialogue) -> str:
    """Format a single dialogue as plain text."""
    lines = []
    lines.append(f"--- Dialogue (seed={dialogue['seed']}) ---")
    lines.append(f"Topic: {dialogue['params']['topic']}")
    lines.append(f"Outcome: {dialogue['params']['outcome']}")
    lines.append("")

    for msg in dialogue["messages"]:
        role_label = "Client" if msg["role"] == "client" else "Agent"
        lines.append(f"{role_label}: {msg['content']}")

    lines.append("")
    return "\n".join(lines)


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base_seed = (
        int(sys.argv[2]) if len(sys.argv) > 2 else random.randint(100000, 999999)
    )

    print(f"Generating {count} non-conflict dialogues...")
    print(f"Base seed: {base_seed}")
    print()

    pipeline = Pipeline()
    dialogues = []
    all_logs = []

    # Find seeds that produce non-conflict outcomes
    seeds = []
    current_seed = base_seed
    for i in range(count):
        seed = generate_non_conflict_seed(current_seed)
        seeds.append(seed)
        current_seed = seed + 100  # offset for next search

    print(f"Selected seeds: {seeds}")
    print()

    for i, seed in enumerate(seeds):
        print(f"=== Generating dialogue {i + 1}/{count} (seed={seed}) ===")

        try:
            result = pipeline.run(seed)

            # Convert to dict
            dialogue_dict = {
                "id": result.id,
                "seed": result.seed,
                "params": {
                    "complexity": result.params.complexity.value
                    if hasattr(result.params.complexity, "value")
                    else str(result.params.complexity),
                    "sector": result.params.sector,
                    "topic": result.params.topic,
                    "outcome": result.params.outcome.value
                    if hasattr(result.params.outcome, "value")
                    else str(result.params.outcome),
                    "style": result.params.style.value
                    if hasattr(result.params.style, "value")
                    else str(result.params.style),
                    "client_archetype": result.params.client_archetype.value
                    if hasattr(result.params.client_archetype, "value")
                    else str(result.params.client_archetype),
                    "agent_archetype": result.params.agent_archetype.value
                    if hasattr(result.params.agent_archetype, "value")
                    else str(result.params.agent_archetype),
                    "twist": result.params.twist,
                    "n_messages": result.params.n_messages,
                },
                "characters": {
                    "client": {
                        "name": result.characters.client.name,
                        "mood": result.characters.client.mood,
                        "personality": result.characters.client.personality,
                        "backstory": result.characters.client.backstory,
                    },
                    "agent": {
                        "name": result.characters.agent.name,
                        "mood": result.characters.agent.mood,
                        "personality": result.characters.agent.personality,
                    },
                },
                "messages": [
                    {
                        "role": msg.role.value
                        if hasattr(msg.role, "value")
                        else str(msg.role),
                        "turn": msg.turn,
                        "content": msg.content,
                        "emotion": msg.emotion,
                        "intensity": msg.intensity,
                    }
                    for msg in result.messages
                ],
                "meta": result.meta,
            }

            dialogues.append(dialogue_dict)

            # Print preview
            print(f"  Topic: {dialogue_dict['params']['topic']}")
            print(f"  Outcome: {dialogue_dict['params']['outcome']}")
            print(f"  Messages: {len(dialogue_dict['messages'])}")
            print()

        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # Generate output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Plain text file with just dialogues
    txt_filename = "dialogues_sample.txt"
    txt_content = []
    txt_content.append(f"Generated: {datetime.now().isoformat()}")
    txt_content.append(f"Count: {len(dialogues)}")
    txt_content.append("=" * 60)
    txt_content.append("")

    for dialogue in dialogues:
        txt_content.append(format_dialogue_text(dialogue))

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_content))

    print(f"Saved plain text: {txt_filename}")

    # 2. Full JSON with all metadata
    json_filename = "dialogues_sample.json"
    json_data = {
        "generated_at": datetime.now().isoformat(),
        "count": len(dialogues),
        "seeds": seeds,
        "dialogues": dialogues,
    }

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Saved JSON: {json_filename}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
