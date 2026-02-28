from config import (
    AGENT_ARCHETYPE_HINTS,
    BACKENDS,
    CLIENT_ARCHETYPE_HINTS,
    MODELS,
    PROMPTS,
)
from core.llm_client import LLMClient, safe_json_parse
from core.models import CharacterProfile, Characters, DiceRollParams


def _val(x) -> str:
    """Return .value if enum, else str."""
    return x.value if hasattr(x, "value") else str(x)


def run(client: LLMClient, params: DiceRollParams) -> Characters:
    client_profile = _generate_character(client, role="client", params=params)
    agent_profile = _generate_character(client, role="agent", params=params)

    if client_profile.name == agent_profile.name:
        agent_profile = agent_profile.model_copy(
            update={"name": agent_profile.name + " (agent)"}
        )

    return Characters(client=client_profile, agent=agent_profile)


def _generate_character(
    client: LLMClient,
    role: str,
    params: DiceRollParams,
) -> CharacterProfile:

    if role == "client":
        hint = CLIENT_ARCHETYPE_HINTS[_val(params.client_archetype)]
        system = PROMPTS["character_client_system"]
        user = PROMPTS["character_client_user"].format(
            client_archetype=_val(params.client_archetype),
            archetype_hint=hint,
            topic=params.topic,
            style=_val(params.style),
        )
    else:
        hint = AGENT_ARCHETYPE_HINTS[_val(params.agent_archetype)]
        system = PROMPTS["character_agent_system"]
        user = PROMPTS["character_agent_user"].format(
            agent_archetype=_val(params.agent_archetype),
            archetype_hint=hint,
        )

    prose = client.complete(
        system=system, user=user, model=MODELS["writer"], backend=BACKENDS["writer"]
    )

    parser_raw = client.complete(
        system=PROMPTS["character_parser_system"],
        user=PROMPTS["character_parser_user"].format(prose=prose),
        model=MODELS["parser"],
        backend=BACKENDS["parser"],
    )

    data = safe_json_parse(parser_raw)
    # Ensure list fields exist
    data.setdefault("personality", [])
    data.setdefault("backstory", "")
    data.setdefault("quirks", [])
    return CharacterProfile(**data)
