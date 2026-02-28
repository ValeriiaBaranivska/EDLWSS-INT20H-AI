"""
debug_pipeline.py — Full debug logging for the entire pipeline.

Runs ONE dialogue with seed=328822 (complex) and ONE with seed=934870 (simple)
and prints EVERYTHING: prompts, raw responses, parsed results, transformations.

Usage:
    python debug_pipeline.py
    python debug_pipeline.py 328822 934870 12345
"""

import random as _random
import re
import sys

from dotenv import load_dotenv

load_dotenv(".env", override=True)

from core.llm_client import LLMClient
from core.models import Role
from pipeline import (
    stage0_dice,
    stage1_characters,
    stage2a_scenes,
    stage2b_voice,
    stage3_emotions,
    stage4_client_style,
    stage4b_agent_style,
    stage5_simple_style,
)
from pipeline.few_shot_bank import FewShotBank
from pipeline.message_plan import plan_to_template
from pipeline.stage6_dirt import DirtLayer


class DebugLLMClient(LLMClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_log = []

    def complete(
        self, system, user, model=None, temperature=0.7, max_tokens=512, backend=None
    ) -> str:

        call_id = len(self.call_log) + 1

        print(f"\n{'=' * 60}")
        print(f"LLM CALL #{call_id}  model={model}  backend={backend}")
        print(f"{'-' * 60}")
        print(f"SYSTEM:\n{system[:500]}")
        if len(system) > 500:
            print(f"  ... [{len(system) - 500} chars truncated]")
        print(f"{'-' * 60}")
        print(f"USER:\n{user[:800]}")
        if len(user) > 800:
            print(f"  ... [{len(user) - 800} chars truncated]")
        print(f"{'-' * 60}")

        result = super().complete(system, user, model, temperature, max_tokens, backend)

        print(f"RAW RESPONSE:\n{result[:600]}")
        if len(result) > 600:
            print(f"  ... [{len(result) - 600} chars truncated]")
        print(f"{'=' * 60}")

        self.call_log.append(
            {
                "id": call_id,
                "model": model,
                "system": system[:200],
                "user": user[:200],
                "result": result[:200],
            }
        )

        return result


def _print_messages(label: str, messages: list) -> None:
    print(f"\n  [{label}] -- {len(messages)} messages:")
    for m in messages:
        role = m.role if isinstance(m.role, str) else m.role.value
        print(f"    turn={m.turn} [{role:6}] {m.content[:70]}")
    # warn on structural issues
    roles = [m.role for m in messages]
    if not messages:
        print("  WARNING: EMPTY -- no messages!")
        return
    if Role.client not in roles:
        print("  WARNING: NO CLIENT MESSAGES")
    if Role.agent not in roles:
        print("  WARNING: NO AGENT MESSAGES")
    prev = None
    run = 0
    for m in messages:
        if m.role == prev:
            run += 1
            if run >= 3:
                print(f"  WARNING: 3+ consecutive {m.role} messages!")
                break
        else:
            run = 1
            prev = m.role


def run_debug(seed: int, label: str):
    print(f"\n{'#' * 70}")
    print(f"# DEBUG RUN: {label}  seed={seed}")
    print(f"{'#' * 70}")

    client = DebugLLMClient()

    # STAGE 0
    print(f"\n{'_' * 40}")
    print("STAGE 0 -- dice_roll")
    print(f"{'_' * 40}")
    params = stage0_dice.run(seed)
    print(f"complexity:       {params.complexity}")
    print(f"n_messages:       {params.n_messages}")
    print(f"outcome:          {params.outcome}")
    print(f"style:            {params.style}")
    print(f"client_archetype: {params.client_archetype}")
    print(f"agent_archetype:  {params.agent_archetype}")
    print(f"twist:            {params.twist}")
    print(f"use_dolphin:      {params.use_dolphin}")

    if params.complexity == "simple":
        # STAGE 5
        print(f"\n{'_' * 40}")
        print("STAGE 5 -- simple generate")
        print(f"{'_' * 40}")
        messages = stage5_simple_style.run(client, params)
        _print_messages("After stage5", messages)

    else:
        # STAGE 1
        print(f"\n{'_' * 40}")
        print("STAGE 1 -- characters")
        print(f"{'_' * 40}")
        characters = stage1_characters.run(client, params)
        print(
            f"client: {characters.client.name} | "
            f"mood={characters.client.mood} | "
            f"backstory={characters.client.backstory[:60] if characters.client.backstory else 'N/A'}"
        )
        print(f"agent:  {characters.agent.name} | mood={characters.agent.mood}")

        # STAGE 2A
        print(f"\n{'_' * 40}")
        print("STAGE 2A -- scene planning")
        print(f"{'_' * 40}")
        rng = _random.Random(params.seed + 200)
        scene_plan, message_plan = stage2a_scenes.run(client, params, characters, rng)
        print(f"message_plan: {message_plan}")
        print(f"n_scenes:     {len(scene_plan.scenes)}")
        print(f"total_msgs:   {scene_plan.total_messages}")
        for s in scene_plan.scenes:
            print(f"\n  Scene {s.id} [{s.beat}]")
            print(f"    expected_messages: {s.expected_messages}")
            desc = s.description[:80] if s.description else "N/A"
            print(f"    description: {desc}")
            print(f"    emotional_state: {s.emotional_state}")

        # STAGE 2B -- show template AND raw voiced per scene
        print(f"\n{'_' * 40}")
        print("STAGE 2B -- voicing (per scene)")
        print(f"{'_' * 40}")

        global_idx = 1
        for scene in scene_plan.scenes:
            local_tpl = plan_to_template(
                scene.expected_messages,
                characters.client.name,
                characters.agent.name,
            )
            lines = local_tpl.split("\n")
            renumbered = [
                re.sub(r"^\d+\.", f"{global_idx + i}.", l) for i, l in enumerate(lines)
            ]

            print(f"\n  -- Scene {scene.id} [{scene.beat}] --")
            print("  Template sent to voice:")
            for l in renumbered:
                print(f"    {l}")

            global_idx += len(scene.expected_messages)

        # now run actual stage2b
        messages = stage2b_voice.run(client, params, characters, scene_plan)
        _print_messages("After stage2b", messages)

        # STAGE 3
        print(f"\n{'_' * 40}")
        print("STAGE 3 -- emotions")
        print(f"{'_' * 40}")
        emotions = stage3_emotions.run(client, messages, params)
        print(f"emotions parsed: {len(emotions.turns)} turns")
        for turn_id, em in emotions.turns.items():
            print(
                f"  turn {turn_id}: "
                f"client={em.client_emotion}({em.client_intensity}) "
                f"agent={em.agent_composure}({em.agent_stress})"
            )

        # STAGE 4
        print(f"\n{'_' * 40}")
        print("STAGE 4 -- client style")
        print(f"{'_' * 40}")
        few_shots = FewShotBank("corpus/bitext_instructions.jsonl")
        messages_before = [m.content[:50] for m in messages if m.role == Role.client]
        messages = stage4_client_style.run(
            client, messages, params, emotions, few_shots
        )
        messages_after = [m.content[:50] for m in messages if m.role == Role.client]
        print("Client messages before/after style:")
        for b, a in zip(messages_before, messages_after):
            print(f"  BEFORE: {b}")
            print(f"  AFTER:  {a}")
            print()

        # STAGE 4B
        print(f"\n{'_' * 40}")
        print("STAGE 4B -- agent style")
        print(f"{'_' * 40}")
        messages_before = [m.content[:50] for m in messages if m.role == Role.agent]
        messages = stage4b_agent_style.run(client, messages, params, emotions)
        messages_after = [m.content[:50] for m in messages if m.role == Role.agent]
        print("Agent messages before/after style:")
        for b, a in zip(messages_before, messages_after):
            print(f"  BEFORE: {b}")
            print(f"  AFTER:  {a}")
            print()

    # STAGE 6
    print(f"\n{'_' * 40}")
    print("STAGE 6 -- dirt layer")
    print(f"{'_' * 40}")
    dirt = DirtLayer(
        slang_bank_path="corpus/slang_bank.json",
        filler_bank_path="corpus/filler_bank.json",
        seed=seed,
    )
    messages_before = [(m.role, m.content[:50]) for m in messages]
    messages, dirt_log = dirt.apply(messages, params)
    print(f"Transforms applied: {dirt_log}")
    print("Messages before/after dirt:")
    for (role, before), msg in zip(messages_before, messages):
        r = role if isinstance(role, str) else role.value
        print(f"  [{r}] BEFORE: {before}")
        print(f"  [{r}] AFTER:  {msg.content[:50]}")
        print()

    _print_messages("FINAL OUTPUT", messages)

    # SUMMARY
    print(f"\n{'=' * 60}")
    print(f"SUMMARY -- seed={seed}")
    print(f"{'=' * 60}")
    print(f"Total LLM calls:  {len(client.call_log)}")
    print(f"Final messages:   {len(messages)}")
    roles = [m.role for m in messages]
    print(f"Client messages:  {roles.count(Role.client)}")
    print(f"Agent messages:   {roles.count(Role.agent)}")
    has_client = Role.client in roles
    has_agent = Role.agent in roles
    if not has_client or not has_agent:
        print("WARNING: missing client or agent messages!")
    elif messages and messages[0].role != Role.client:
        print("WARNING: first message is not client!")
    else:
        print("OK: Structure OK")


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == "__main__":
    seeds = [328822, 934870]
    if len(sys.argv) > 1:
        seeds = [int(x) for x in sys.argv[1:]]

    log_path = "debug_pipeline_out.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        try:
            for seed in seeds:
                # determine complexity by running dice roll
                test_params = stage0_dice.run(seed)
                label = "COMPLEX" if test_params.complexity == "complex" else "SIMPLE"
                run_debug(seed, label)
        finally:
            sys.stdout = sys.__stdout__

    print(f"\nFull log saved to: {log_path}")
