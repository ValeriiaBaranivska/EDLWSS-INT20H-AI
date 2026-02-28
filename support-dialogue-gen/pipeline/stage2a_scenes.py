"""stage2a_scenes.py — Director: llama writes a structured scene plan.

Returns (ScenePlan, message_plan) where message_plan is the flat role list
used by stage2b_voice to fill in each message slot.
"""

import random

from config import (
    AGENT_ARCHETYPE_HINTS,
    BACKENDS,
    CLIENT_ARCHETYPE_HINTS,
    MODELS,
    OUTCOME_INSTRUCTIONS,
    PROMPTS,
)
from core.llm_client import LLMClient, safe_json_parse
from core.models import Characters, DiceRollParams, Scene, ScenePlan

from pipeline.message_plan import generate_plan, plan_to_template


def _val(x) -> str:
    """Return .value if enum, else str."""
    return x.value if hasattr(x, "value") else str(x)


def _normalize_expected_messages(raw: list) -> list[str]:
    """
    Convert whatever llama returns into clean ["client","agent"] list.
    Handles:
      - already correct: ["client", "agent"] -> keep
      - full text: ["CLIENT (Aurora): ...", "AGENT (Maya): ..."]
                   -> extract role from prefix
      - mixed: some correct, some full text -> normalize all
    """
    result = []
    for item in raw:
        if not isinstance(item, str):
            result.append("client")
            continue
        s = item.strip().lower()
        if s in ("client", "agent"):
            result.append(s)
        elif s.startswith("client") or s.startswith("customer"):
            result.append("client")
        elif s.startswith("agent"):
            result.append("agent")
        else:
            # unknown — alternate client/agent based on position
            result.append("client" if len(result) % 2 == 0 else "agent")
    return result


def run(
    client: LLMClient,
    params: DiceRollParams,
    characters: Characters,
    rng: random.Random,
) -> tuple[ScenePlan, list[str]]:
    """Returns (ScenePlan, message_plan)."""

    message_plan = generate_plan(
        n_messages=params.n_messages,
        outcome=params.outcome,
        rng=rng,
    )

    template = plan_to_template(
        message_plan,
        characters.client.name,
        characters.agent.name,
    )

    n_scenes = max(2, params.n_messages // 3)

    twist_line = (
        "No special twist needed."
        if params.twist == "none"
        else f"REQUIRED TWIST — must appear in one scene beat: {params.twist}"
    )

    prose = client.complete(
        system=PROMPTS["scene_writer_system"],
        user=PROMPTS["scene_writer_user"].format(
            topic=params.topic,
            sector=params.sector,
            client_name=characters.client.name,
            client_hint=CLIENT_ARCHETYPE_HINTS[_val(params.client_archetype)],
            client_mood=characters.client.mood,
            agent_name=characters.agent.name,
            agent_hint=AGENT_ARCHETYPE_HINTS[_val(params.agent_archetype)],
            outcome_instruction=OUTCOME_INSTRUCTIONS[_val(params.outcome)],
            twist_line=twist_line,
            emotional_arc=_val(params.emotional_arc),
            n_messages=params.n_messages,
            message_plan=template,
            n_scenes=n_scenes,
        ),
        model=MODELS["writer"],
        max_tokens=1024,
        backend=BACKENDS["writer"],
    )

    data = safe_json_parse(prose)
    scenes_raw = data.get("scenes", data) if isinstance(data, dict) else data

    scenes = [Scene(**s) for s in scenes_raw]

    # Normalize expected_messages — llama may return full dialogue text instead of roles
    for scene in scenes:
        scene.expected_messages = _normalize_expected_messages(scene.expected_messages)

    # Validate: total expected_messages must match n_messages
    total = sum(len(s.expected_messages) for s in scenes)
    if total != params.n_messages:
        _fix_scene_distribution(scenes, message_plan)

    scene_plan = ScenePlan(
        scenes=scenes,
        total_messages=params.n_messages,
    )

    return scene_plan, message_plan


def _fix_scene_distribution(
    scenes: list[Scene],
    message_plan: list[str],
) -> None:
    """Redistribute message_plan across scenes if counts mismatch.
    Mutates scenes in place."""
    idx = 0
    for scene in scenes:
        count = max(1, len(scene.expected_messages))
        scene.expected_messages = message_plan[idx : idx + count]
        idx += count
    # assign any leftover to last scene
    if idx < len(message_plan) and scenes:
        scenes[-1].expected_messages += message_plan[idx:]
