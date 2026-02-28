from config import AGENT_ARCHETYPE_HINTS, BACKENDS, MODELS, PROMPTS
from core.llm_client import LLMClient
from core.models import DiceRollParams, EmotionAnalysis, Message, Role


def _val(x) -> str:
    """Return .value if enum, else str."""
    return x.value if hasattr(x, "value") else str(x)


def _is_mostly_english(text: str) -> bool:
    """Return True if text is mostly ASCII (English)."""
    if not text:
        return True
    ascii_count = sum(1 for c in text if c.isascii())
    return ascii_count / len(text) > 0.85


def _clean_output(text: str, original: str) -> str:
    """Clean LLM output: strip quotes, remove meta-leaks, validate ending.
    Falls back to original if output is invalid."""
    if not text or not text.strip():
        return original

    result = text.strip()

    # Strip dolphin's quote wrapping and escaped quotes
    result = result.strip("\"'")
    result = result.replace('\\"', '"').replace("\\'", "'")

    # Remove common meta-leaks / hallucinated suffixes
    meta_patterns = [
        "ispersumably",
        "isper sumably",
        "is presumably",
        "(rewritten)",
        "(in character)",
        "[rewritten]",
        "---",
        "...",
    ]
    for pattern in meta_patterns:
        if result.lower().endswith(pattern):
            result = result[: -len(pattern)].rstrip()

    # Ensure proper ending punctuation
    if result and result[-1] not in ".!?":
        # Check if it looks truncated (ends mid-word)
        if len(result) > 5 and result[-1].isalpha():
            result = result + "."

    # English guard
    if not _is_mostly_english(result):
        return original

    # Length sanity check
    if len(result) < 3 or len(result) > 500:
        return original

    return result


def run(
    client: LLMClient,
    messages: list[Message],
    params: DiceRollParams,
    emotions: EmotionAnalysis,
) -> list[Message]:

    model = MODELS["dolphin"] if params.use_dolphin else MODELS["styler"]
    model_backend = BACKENDS["dolphin"] if params.use_dolphin else BACKENDS["styler"]

    result = []

    for idx, msg in enumerate(messages):
        if msg.role != Role.agent:
            result.append(msg)
            continue

        if not msg.content.strip():
            result.append(msg)
            continue

        turn_data = emotions.turns.get(str(msg.turn))
        composure = turn_data.agent_composure if turn_data else "steady"
        stress = turn_data.agent_stress if turn_data else 2

        # Find the most recent client message before this agent message
        prev_client = ""
        for i in range(idx - 1, -1, -1):
            if messages[i].role == Role.client:
                prev_client = messages[i].content
                break

        twist_context = ""
        if composure == "lost" and params.twist == "agent_breaks_protocol":
            twist_context = (
                "You are about to break protocol. "
                "Say something you shouldn't. Once. Briefly."
            )
        elif composure == "slipping":
            twist_context = "You are losing patience. Be slightly sharp or dry."

        agent_key = _val(params.agent_archetype)
        rewritten = client.complete(
            system=PROMPTS["agent_style_system"].format(
                agent_name="Agent",
                agent_hint=AGENT_ARCHETYPE_HINTS[agent_key],
                composure=composure,
                stress=stress,
                twist_context=twist_context,
            ),
            user=PROMPTS["agent_style_user"].format(
                client_message=prev_client,
                original_agent=msg.content,
            ),
            model=model,
            max_tokens=150,
            backend=model_backend,
        )

        # Clean output: strip quotes, meta-leaks, validate
        cleaned = _clean_output(rewritten, msg.content)

        if cleaned.strip() and cleaned.strip() != msg.content:
            result.append(msg.model_copy(update={"content": cleaned.strip()}))
        else:
            result.append(msg)

    return result
