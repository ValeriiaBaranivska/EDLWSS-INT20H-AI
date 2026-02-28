from config import (
    AGENT_ARCHETYPE_HINTS,
    BACKENDS,
    CLIENT_ARCHETYPE_HINTS,
    MODELS,
    OUTCOME_INSTRUCTIONS,
    PROMPTS,
)
from core.llm_client import LLMClient, safe_json_parse
from core.models import Characters, DiceRollParams, Message, Role


def run(
    client: LLMClient,
    params: DiceRollParams,
    characters: Characters,
) -> list[Message]:

    # Support both old target_turns and new n_messages
    target_turns = getattr(params, "target_turns", None) or params.n_messages

    twist_line = (
        "No special twist needed."
        if params.twist == "none"
        else f"REQUIRED TWIST — must be visible in text: {params.twist}"
    )

    outcome_instruction = OUTCOME_INSTRUCTIONS[params.outcome]

    prose = client.complete(
        system=PROMPTS["story_writer_system"].format(
            target_turns=target_turns,
        ),
        user=PROMPTS["story_writer_user"].format(
            topic=params.topic,
            sector=params.sector,
            client_name=characters.client.name,
            client_hint=CLIENT_ARCHETYPE_HINTS[params.client_archetype],
            client_mood=characters.client.mood,
            client_backstory=characters.client.backstory,
            agent_name=characters.agent.name,
            agent_hint=AGENT_ARCHETYPE_HINTS[params.agent_archetype],
            outcome_instruction=outcome_instruction,
            twist_line=twist_line,
            emotional_arc=params.emotional_arc,
            target_turns=target_turns,
        ),
        model=MODELS["writer"],
        max_tokens=2048,
        backend=BACKENDS["writer"],
    )

    parser_raw = client.complete(
        system=PROMPTS["story_parser_system"],
        user=PROMPTS["story_parser_user"].format(
            raw_story=prose,
            client_name=characters.client.name,
            agent_name=characters.agent.name,
        ),
        model=MODELS["parser"],
        max_tokens=1024,
        backend=BACKENDS["parser"],
    )

    data = safe_json_parse(parser_raw)
    # data may be a list directly, or a dict with a key
    if isinstance(data, dict):
        data = data.get("messages", data.get("dialogue", list(data.values())[0]))

    messages = [Message(**m) for m in data]
    messages = _trim_to_turns(messages, target_turns)
    messages = _deduplicate(messages)
    messages = _fix_turn_numbers(messages)
    messages = _fix_ending(messages, params.outcome)
    return messages


# ── helpers ──────────────────────────────────────────────────


def _trim_to_turns(messages: list[Message], n: int) -> list[Message]:
    result, count = [], 0
    for msg in messages:
        if msg.role == Role.client:
            count += 1
        if count > n:
            break
        result.append(msg)
    return result


def _deduplicate(messages: list[Message]) -> list[Message]:
    seen, result = set(), []
    for msg in messages:
        key = (msg.role, msg.content.strip()[:80])
        if key not in seen:
            seen.add(key)
            result.append(msg)
    return result


def _fix_turn_numbers(messages: list[Message]) -> list[Message]:
    """Enforce turn numbering: client+agent pairs share the same turn number.
    Handles models that output 1,2,3,4 instead of 1,1,2,2."""
    result = []
    turn = 0
    for msg in messages:
        if msg.role == Role.client:
            turn += 1
        result.append(msg.model_copy(update={"turn": turn}))
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
        if messages[-1].role == Role.agent:
            messages = messages[:-1]
        if messages:
            last = messages[-1].content.lower()
            if not any(p in last for p in EXIT_PHRASES):
                messages[-1] = messages[-1].model_copy(
                    update={"content": messages[-1].content + " Forget it."}
                )
    return messages
