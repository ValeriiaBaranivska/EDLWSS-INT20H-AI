from config import BACKENDS, MODELS, PROMPTS
from core.llm_client import LLMClient, safe_json_parse
from core.models import DiceRollParams, EmotionAnalysis, EmotionTurn, Message


def run(
    client: LLMClient,
    messages: list[Message],
    params: DiceRollParams,
) -> EmotionAnalysis:
    n_turns = max(m.turn for m in messages)
    dialogue_text = _format_messages(messages)

    prose = client.complete(
        system=PROMPTS["emotion_writer_system"],
        user=PROMPTS["emotion_writer_user"].format(
            dialogue_text=dialogue_text,
            n_turns=n_turns,
        ),
        model=MODELS["writer"],
        backend=BACKENDS["writer"],
    )

    parser_raw = client.complete(
        system=PROMPTS["emotion_parser_system"],
        user=PROMPTS["emotion_parser_user"].format(prose=prose),
        model=MODELS["parser"],
        max_tokens=1024,  # Increased for long dialogues (16+ messages)
        backend=BACKENDS["parser"],
    )

    data = safe_json_parse(parser_raw)
    turns_raw = data.get("turns", data)

    turns = {}
    for k, v in turns_raw.items():
        turns[str(k)] = EmotionTurn(**_normalize_emotion_turn(v))

    return EmotionAnalysis(turns=turns)


def _normalize_emotion_turn(d: dict) -> dict:
    """
    Llama/qwen sometimes return non-standard field names.
    Normalize to EmotionTurn schema:
      client_emotion, client_intensity,
      agent_composure, agent_stress
    """
    result = {}

    # ── client_emotion ──────────────────────────────────────
    if "client_emotion" in d:
        result["client_emotion"] = d["client_emotion"]
    else:
        # scan for any key containing "emotion" or "feeling"
        for key in d:
            if "emotion" in key.lower() or "feeling" in key.lower():
                result["client_emotion"] = str(d[key])
                break
        else:
            result["client_emotion"] = "neutral"

    # ── client_intensity ────────────────────────────────────
    if "client_intensity" in d:
        result["client_intensity"] = d["client_intensity"]
    else:
        # scan for key containing "intensity" or "level"
        for key in d:
            if (
                "intensity" in key.lower()
                or "level" in key.lower()
                or "score" in key.lower()
            ):
                try:
                    result["client_intensity"] = int(d[key])
                except (ValueError, TypeError):
                    result["client_intensity"] = 3
                break
        else:
            result["client_intensity"] = 3

    # ── agent_composure ─────────────────────────────────────
    VALID_COMPOSURE = {"steady", "holding", "slipping", "lost"}
    if "agent_composure" in d:
        result["agent_composure"] = d["agent_composure"]
    else:
        for key in d:
            if "composure" in key.lower() or "agent" in key.lower():
                val = str(d[key]).lower().strip()
                if val in VALID_COMPOSURE:
                    result["agent_composure"] = val
                    break
        else:
            result["agent_composure"] = "steady"

    # ensure valid value
    if result.get("agent_composure") not in VALID_COMPOSURE:
        result["agent_composure"] = "steady"

    # ── agent_stress ────────────────────────────────────────
    if "agent_stress" in d:
        result["agent_stress"] = d["agent_stress"]
    else:
        for key in d:
            if "stress" in key.lower():
                try:
                    result["agent_stress"] = int(d[key])
                except (ValueError, TypeError):
                    result["agent_stress"] = 2
                break
        else:
            result["agent_stress"] = 2

    return result


def _format_messages(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        role_label = "Client" if m.role == "client" else "Agent"
        lines.append(f"[Turn {m.turn}] {role_label}: {m.content}")
    return "\n".join(lines)
