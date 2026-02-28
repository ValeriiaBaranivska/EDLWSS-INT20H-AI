"""stage2b_voice.py — Voice actor: qwen/dolphin fills each scene template.

Processes each scene from ScenePlan independently, then merges into a flat
Message list with correct paired turn numbers.
"""

import re

from config import (
    AGENT_ARCHETYPE_HINTS,
    BACKENDS,
    CLIENT_ARCHETYPE_HINTS,
    MODELS,
    PROMPTS,
)
from core.llm_client import LLMClient
from core.models import Characters, DiceRollParams, Message, Role, ScenePlan

from pipeline.message_plan import plan_to_template


def _is_mostly_english(text: str) -> bool:
    """Return True if text is mostly ASCII (English)."""
    if not text:
        return True
    ascii_count = sum(1 for c in text if c.isascii())
    return ascii_count / len(text) > 0.85


def run(
    client: LLMClient,
    params: DiceRollParams,
    characters: Characters,
    scene_plan: ScenePlan,
) -> list[Message]:
    """Voice each scene, return flat Message list."""
    all_messages: list[Message] = []
    global_idx = 1  # running message number across all scenes

    for scene in scene_plan.scenes:
        # Build numbered template for this scene, renumbered from global_idx
        local_template = plan_to_template(
            scene.expected_messages,
            characters.client.name,
            characters.agent.name,
        )
        lines = local_template.split("\n")
        renumbered = []
        for i, line in enumerate(lines):
            renumbered.append(re.sub(r"^\d+\.", f"{global_idx + i}.", line))
        template = "\n".join(renumbered)

        # Use dolphin for conflict/escalation scenes, styler otherwise
        is_conflict = any(
            word in scene.beat.lower()
            for word in (
                "conflict",
                "ragequit",
                "escalat",
                "twist",
                "protocol",
                "legal",
            )
        )
        model = (
            MODELS["dolphin"]
            if (params.use_dolphin and is_conflict)
            else MODELS["styler"]
        )
        backend = (
            BACKENDS["dolphin"]
            if (params.use_dolphin and is_conflict)
            else BACKENDS["styler"]
        )

        voiced = client.complete(
            system=PROMPTS["voice_writer_system"],
            user=PROMPTS["voice_writer_user"].format(
                beat=scene.beat,
                description=scene.description,
                client_goal=scene.client_goal,
                agent_goal=scene.agent_goal,
                emotional_state=scene.emotional_state,
                client_name=characters.client.name,
                client_hint=CLIENT_ARCHETYPE_HINTS[params.client_archetype],
                agent_name=characters.agent.name,
                agent_hint=AGENT_ARCHETYPE_HINTS[params.agent_archetype],
                template=template,
            ),
            model=model,
            max_tokens=512,
            backend=backend,
        )

        messages = _parse_voiced(
            voiced,
            scene.expected_messages,
            characters.client.name,
            characters.agent.name,
            global_idx,
        )
        all_messages.extend(messages)
        global_idx += len(scene.expected_messages)

    all_messages = _fix_turn_numbers(all_messages)
    all_messages = _deduplicate(all_messages)
    all_messages = _fix_consecutive_agents(all_messages)
    all_messages = _fix_ending(all_messages, params.outcome)
    return all_messages


# ── parsers + helpers ──────────────────────────────────────────────────────


def _parse_voiced(
    text: str,
    plan: list[str],
    client_name: str,
    agent_name: str,
    start_idx: int,
) -> list[Message]:
    """Parse numbered lines from voiced output back into Message objects."""
    messages: list[Message] = []
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    for i, role in enumerate(plan):
        content = ""
        expected_num = start_idx + i

        # Find matching numbered line
        for line in lines:
            if re.match(rf"^{expected_num}\.", line):
                # Strip "N. ROLE (Name): " prefix — use .*? to handle nested parens
                content = re.sub(
                    rf"^{expected_num}\.\s*(CLIENT|AGENT)\s*\(.*?\):\s*",
                    "",
                    line,
                    flags=re.IGNORECASE,
                ).strip()
                if not content:
                    content = re.sub(rf"^{expected_num}\.\s*", "", line).strip()
                # additional fallback: strip anything before "): "
                if "):" in content:
                    content = content.split("):", 1)[-1].strip()
                break

        if not content and lines:
            # Fallback: take next available line by index
            fallback = lines[i] if i < len(lines) else f"[{role}]"
            content = re.sub(r"^\d+\.\s*", "", fallback).strip()

        # Strip dolphin's quote wrapping and escaped quotes
        content = content.strip("\"'")
        content = content.replace('\\"', '"').replace("\\'", "'")

        # English guard: fallback if too many non-ASCII chars
        if not _is_mostly_english(content):
            content = f"[{role}]"

        messages.append(
            Message(
                role=Role.client if role == "client" else Role.agent,
                turn=0,  # fixed by _fix_turn_numbers
                content=content or f"[{role}]",
            )
        )

    return messages


def _fix_turn_numbers(messages: list[Message]) -> list[Message]:
    """Assign sequential turn numbers: turn = 1-based position in message list.

    After the n_messages refactor, turn is simply the ordinal position,
    not tied to client/agent exchange pairs anymore.
    """
    return [msg.model_copy(update={"turn": i + 1}) for i, msg in enumerate(messages)]


def _fix_consecutive_agents(messages: list[Message]) -> list[Message]:
    """Remove duplicate consecutive agent messages at the end.

    The scene generator sometimes produces two agent messages at the end
    when the message_plan already ends with 'agent'.
    """
    while (
        len(messages) >= 2
        and messages[-1].role == Role.agent
        and messages[-2].role == Role.agent
    ):
        messages = messages[:-1]
    return messages


def _deduplicate(messages: list[Message]) -> list[Message]:
    seen, result = set(), []
    for msg in messages:
        key = (msg.role, msg.content.strip()[:80])
        if key not in seen:
            seen.add(key)
            result.append(msg)
    return result


EXIT_PHRASES = [
    "forget it",
    "forget this",
    "i'm done",
    "this is useless",
    "goodbye",
    "unbelievable",
    "waste of time",
]


def _fix_ending(messages: list[Message], outcome: str) -> list[Message]:
    if not messages:
        return messages
    if outcome == "unresolved_ragequit":
        # last message must be from client
        if messages[-1].role == Role.agent:
            messages = messages[:-1]
        if messages:
            last = messages[-1].content.lower()
            if not any(p in last for p in EXIT_PHRASES):
                messages[-1] = messages[-1].model_copy(
                    update={"content": messages[-1].content + " Forget it."}
                )
    return messages
