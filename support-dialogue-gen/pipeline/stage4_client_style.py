from config import BACKENDS, MODELS, PROMPTS
from core.llm_client import LLMClient
from core.models import DiceRollParams, EmotionAnalysis, Message, Role


def _val(x) -> str:
    """Return .value if enum, else str."""
    return x.value if hasattr(x, "value") else str(x)


ARCHETYPE_TO_STYLE: dict[str, str] = {
    "karen": "aggressive",
    "angry_veteran": "aggressive",
    "calling_bluff": "aggressive",
    "elderly_confused": "casual",
    "conspirologist": "passive_aggressive",
    "young_professional": "casual",
    "tech_confused": "casual",
    "grieving": "casual",
    "self_inflicted": "casual",
    "entitled_parent": "aggressive",
    "wrong_department": "casual",
}

_BOT_STARTS = [
    "i'd be happy",
    "certainly",
    "of course",
    "sure!",
    "thank you for",
    "i apologize",
    "i'm sorry",
    "i understand your frustration",
    "i know how frustrating",
    "that must be",
    "absolutely",
]


def run(
    client: LLMClient,
    messages: list[Message],
    params: DiceRollParams,
    emotions: EmotionAnalysis,
    few_shots: dict,
) -> list[Message]:

    archetype_key = _val(params.client_archetype)
    style_key = _val(params.style)
    style = ARCHETYPE_TO_STYLE.get(archetype_key, style_key)
    result = []

    for msg in messages:
        if msg.role != Role.client:
            result.append(msg)
            continue

        if not msg.content.strip():
            result.append(msg)
            continue

        turn_data = emotions.turns.get(str(msg.turn))
        emotion = turn_data.client_emotion if turn_data else "neutral"
        intensity = turn_data.client_intensity if turn_data else 3

        examples = few_shots.bank.get(style, [])[:3]
        ex1 = examples[0] if len(examples) > 0 else "example not available"
        ex2 = examples[1] if len(examples) > 1 else "example not available"
        ex3 = examples[2] if len(examples) > 2 else "example not available"

        rewritten = client.complete(
            system=PROMPTS["client_style_system"],
            user=PROMPTS["client_style_user"].format(
                style=style,
                emotion=emotion,
                intensity=intensity,
                ex1=ex1,
                ex2=ex2,
                ex3=ex3,
                original=msg.content,
            ),
            model=MODELS["styler"],
            max_tokens=150,
            backend=BACKENDS["styler"],
        )

        if _is_valid_rewrite(rewritten, msg.content):
            result.append(
                msg.model_copy(
                    update={
                        "content": rewritten.strip(),
                        "emotion": emotion,
                        "intensity": intensity,
                    }
                )
            )
        else:
            result.append(msg)

    return result


def _is_valid_rewrite(rewritten: str, original: str) -> bool:
    r = rewritten.strip()
    if len(r) < 5 or len(r.split()) > 80:
        return False
    if r.lower() == original.lower().strip():
        return False
    if any(r.lower().startswith(p) for p in _BOT_STARTS):
        return False
    return True
