"""debug_run.py — Full pipeline trace, writes structured JSON to output/.

Usage:
  python debug_run.py               # seed=12345
  python debug_run.py --seed 42

Output: output/debug_<seed>.json
Console: one progress line per stage only.
"""

from dotenv import load_dotenv

load_dotenv(".env", override=True)

import argparse
import json
import random
import time
from pathlib import Path

from core.llm_client import LLMClient
from core.models import CharacterProfile, Characters


def _msgs(messages: list) -> list[dict]:
    return [
        {
            "turn": m.turn,
            "role": m.role.value if hasattr(m.role, "value") else str(m.role),
            "content": m.content,
        }
        for m in messages
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    seed = args.seed

    out_path = Path("output") / f"debug_{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {"seed": seed, "stages": {}}
    client = LLMClient()

    def tick(label: str) -> float:
        print(f"  [{label}] ...", flush=True)
        return time.time()

    def tock(label: str, t0: float, extra: dict | None = None):
        elapsed = round(time.time() - t0, 2)
        stage = {"elapsed_s": elapsed}
        if extra:
            stage.update(extra)
        report["stages"][label] = stage
        print(f"  [{label}] done in {elapsed}s", flush=True)

    # ── Stage 0 ─────────────────────────────────────────────
    from pipeline import stage0_dice

    t0 = tick("stage0_dice")
    params = stage0_dice.run(seed)
    tock("stage0_dice", t0, {"params": params.model_dump()})

    # ── Stage 1 ─────────────────────────────────────────────
    if params.complexity == "complex":
        from pipeline import stage1_characters

        t0 = tick("stage1_characters")
        characters = stage1_characters.run(client, params)
        tock(
            "stage1_characters",
            t0,
            {
                "client": characters.client.model_dump(),
                "agent": characters.agent.model_dump(),
            },
        )
    else:
        characters = Characters(
            client=CharacterProfile(
                name="Customer", mood="neutral", personality=["casual"]
            ),
            agent=CharacterProfile(
                name="Agent", mood="neutral", personality=["professional"], quirks=[]
            ),
        )
        report["stages"]["stage1_characters"] = {
            "skipped": True,
            "reason": "simple complexity",
        }
        print("  [stage1_characters] skipped (simple)", flush=True)

    # ── Stage 2a ────────────────────────────────────────────
    from pipeline import stage2a_scenes

    rng2 = random.Random(params.seed + 200)
    t0 = tick("stage2a_scenes")
    scene_plan, message_plan = stage2a_scenes.run(client, params, characters, rng2)
    tock(
        "stage2a_scenes",
        t0,
        {
            "message_plan": message_plan,
            "total_messages": scene_plan.total_messages,
            "scenes": [
                {
                    "id": s.id,
                    "beat": s.beat,
                    "description": s.description,
                    "client_goal": s.client_goal,
                    "agent_goal": s.agent_goal,
                    "emotional_state": s.emotional_state,
                    "expected_messages": s.expected_messages,
                }
                for s in scene_plan.scenes
            ],
        },
    )

    # ── Stage 2b ────────────────────────────────────────────
    from pipeline import stage2b_voice

    t0 = tick("stage2b_voice")
    messages = stage2b_voice.run(client, params, characters, scene_plan)
    tock("stage2b_voice", t0, {"messages": _msgs(messages)})

    # ── Stage 3 / 4 / 4B  or  5 ────────────────────────────
    if params.complexity == "complex":
        from pipeline import stage3_emotions, stage4_client_style, stage4b_agent_style
        from pipeline.few_shot_bank import FewShotBank

        few_shots = FewShotBank("corpus/bitext_instructions.jsonl")

        t0 = tick("stage3_emotions")
        emotions = stage3_emotions.run(client, messages, params)
        tock(
            "stage3_emotions",
            t0,
            {
                "turns": {
                    k: {
                        "client_emotion": v.client_emotion,
                        "client_intensity": v.client_intensity,
                        "agent_composure": v.agent_composure,
                        "agent_stress": v.agent_stress,
                    }
                    for k, v in emotions.turns.items()
                }
            },
        )

        t0 = tick("stage4_client_style")
        before4 = _msgs(messages)
        messages = stage4_client_style.run(
            client, messages, params, emotions, few_shots
        )
        tock("stage4_client_style", t0, {"before": before4, "after": _msgs(messages)})

        t0 = tick("stage4b_agent_style")
        before4b = _msgs(messages)
        messages = stage4b_agent_style.run(client, messages, params, emotions)
        tock("stage4b_agent_style", t0, {"before": before4b, "after": _msgs(messages)})
    else:
        from pipeline import stage5_simple_style

        t0 = tick("stage5_simple_style")
        before5 = _msgs(messages)
        messages = stage5_simple_style.run(client, messages, params)
        tock("stage5_simple_style", t0, {"before": before5, "after": _msgs(messages)})

    # ── Stage 6 ─────────────────────────────────────────────
    from pipeline import stage6_dirt

    t0 = tick("stage6_dirt")
    before6 = _msgs(messages)
    messages, dirt_applied = stage6_dirt.DirtLayer(
        slang_bank_path="corpus/slang_bank.json",
        filler_bank_path="corpus/filler_bank.json",
        seed=params.seed,
    ).apply(messages, params)
    tock(
        "stage6_dirt",
        t0,
        {
            "dirt_applied": dirt_applied,
            "before": before6,
            "after": _msgs(messages),
        },
    )

    # ── Summary ─────────────────────────────────────────────
    diff = abs(len(messages) - params.n_messages)
    report["summary"] = {
        "n_messages_param": params.n_messages,
        "actual_messages": len(messages),
        "match": "OK" if diff <= 1 else f"DIFF={diff}",
        "final_messages": _msgs(messages),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"  n_messages param : {params.n_messages}")
    print(f"  actual messages  : {len(messages)}")
    print(f"  match            : {report['summary']['match']}")
    print(f"  report saved  -> {out_path}")


if __name__ == "__main__":
    main()
