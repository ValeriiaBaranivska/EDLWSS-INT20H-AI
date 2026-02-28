from config import (
    AGENT_ARCHETYPE_HINTS,
    BACKENDS,
    CLIENT_ARCHETYPE_HINTS,
    MODELS,
    OUTCOME_INSTRUCTIONS,
)
from core.llm_client import LLMClient
from core.models import Characters, DiceRollParams, Message, Role


def _val(x) -> str:
    """Return .value if enum, else str."""
    return x.value if hasattr(x, "value") else str(x)


def run(
    client: LLMClient,
    params: DiceRollParams,
    characters: Characters | None = None,
) -> list[Message]:
    """Simple path: generate dialogue from scratch in one shot."""

    outcome_instruction = OUTCOME_INSTRUCTIONS[params.outcome]
    client_hint = CLIENT_ARCHETYPE_HINTS[params.client_archetype]
    agent_hint = AGENT_ARCHETYPE_HINTS[params.agent_archetype]

    twist_line = (
        "No special twist." if params.twist == "none" else f"Twist: {params.twist}"
    )

    # Extract persona context from characters
    if characters:
        client_mood = characters.client.mood
        client_personality = (
            ", ".join(characters.client.personality)
            if characters.client.personality
            else "casual"
        )
        agent_personality = (
            ", ".join(characters.agent.personality)
            if characters.agent.personality
            else "professional"
        )
        agent_quirks = (
            ", ".join(characters.agent.quirks) if characters.agent.quirks else "none"
        )
        persona_context = (
            f"CLIENT persona: mood={client_mood}, personality={client_personality}\n"
            f"AGENT persona: personality={agent_personality}, quirks={agent_quirks}\n"
        )
    else:
        persona_context = ""

    # build numbered template so model knows exact structure
    plan = _simple_plan(params.n_messages, params.outcome)
    template_lines = []
    for i, role in enumerate(plan):
        name = "Customer" if role == "client" else "Agent"
        template_lines.append(f"{i + 1}. {name}: [message]")
    template = "\n".join(template_lines)

    style_val = _val(params.style)
    twist_val = _val(params.twist) if params.twist != "none" else "none"

    prose = client.complete(
        system=(
            "You write short customer support CHAT dialogues.\n"
            "Reply ONLY in English.\n"
            "CLIENT: 1-2 sentences, can be emotional.\n"
            "AGENT: MAX 1 sentence, 15 words or less. Chat style, not email.\n"
            "Fill in the template exactly.\n"
            "Output ONLY the filled template, nothing else.\n\n"
            "FORBIDDEN phrases (NEVER use):\n"
            "- 'That, ' at start of any message\n"
            "- 'Bear with me'\n"
            "- 'Sorry, got another chat'\n"
            "- 'One moment —'\n"
            "- 'Let me check this for you'\n"
            "Write naturally, each reply should be UNIQUE."
        ),
        user=(
            f"Topic: {params.topic}\n"
            f"Sector: {params.sector}\n"
            f"{persona_context}"
            f"Client style: {style_val} — {client_hint}\n"
            f"Agent type: {agent_hint}\n"
            f"Outcome: {outcome_instruction}\n"
            f"{twist_line}\n\n"
            f"Fill in:\n{template}"
        ),
        model=MODELS["styler"],
        max_tokens=512,
        backend=BACKENDS["styler"],
    )

    messages = _parse(prose, plan)

    if len(messages) < 2:
        messages = _fallback(params)

    return messages


def _simple_plan(n: int, outcome: str) -> list[str]:
    LAST = {
        "unresolved_ragequit": "client",
        "unresolved_passive": "client",
        "resolved_quick": "agent",
        "resolved_neutral": "agent",
        "conflict": "agent",
        "info_only": "agent",
    }
    last = LAST.get(outcome, "agent")
    if n <= 2:
        return ["client", last]
    middle = []
    toggle = "agent"
    for _ in range(n - 2):
        middle.append(toggle)
        toggle = "client" if toggle == "agent" else "agent"
    return ["client"] + middle + [last]


def _parse(text: str, plan: list[str]) -> list[Message]:
    import re

    messages = []
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for i, role in enumerate(plan):
        num = i + 1
        content = ""
        for line in lines:
            if re.match(rf"^{num}\.", line):
                # strip "N. Name: "
                content = re.sub(r"^\d+\.\s*\w+:\s*", "", line).strip()
                # fallback strip
                if not content:
                    content = re.sub(r"^\d+\.\s*", "", line).strip()
                break
        if not content and i < len(lines):
            content = re.sub(r"^\d+\.\s*\w*:?\s*", "", lines[i]).strip()
        if content:
            messages.append(
                Message(
                    role=Role.client if role == "client" else Role.agent,
                    turn=0,  # fixed below
                    content=content,
                )
            )
    # fix turn numbers: sequential 1-based position
    return [m.model_copy(update={"turn": i + 1}) for i, m in enumerate(messages)]


def _fallback(params: DiceRollParams) -> list[Message]:
    return [
        Message(
            role=Role.client,
            turn=1,
            content=f"Hi, I have an issue with {params.topic}.",
        ),
        Message(role=Role.agent, turn=1, content="Let me look into that for you."),
    ]
