from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

# ── Enums ────────────────────────────────────────────────────


class Complexity(str, Enum):
    simple = "simple"
    complex = "complex"


class Style(str, Enum):
    casual = "casual"
    aggressive = "aggressive"
    formal = "formal"
    passive_aggressive = "passive_aggressive"


class Outcome(str, Enum):
    resolved_quick = "resolved_quick"
    resolved_neutral = "resolved_neutral"
    unresolved_passive = "unresolved_passive"
    unresolved_ragequit = "unresolved_ragequit"
    conflict = "conflict"
    info_only = "info_only"


class ClientArchetype(str, Enum):
    karen = "karen"
    angry_veteran = "angry_veteran"
    elderly_confused = "elderly_confused"
    tech_confused = "tech_confused"
    young_professional = "young_professional"
    self_inflicted = "self_inflicted"
    conspirologist = "conspirologist"
    grieving = "grieving"
    wrong_department = "wrong_department"
    calling_bluff = "calling_bluff"
    entitled_parent = "entitled_parent"


class AgentArchetype(str, Enum):
    veteran_tired = "veteran_tired"
    burned_out = "burned_out"
    hands_tied = "hands_tied"
    eager_helper = "eager_helper"
    by_the_book = "by_the_book"
    newbie_overwhelmed = "newbie_overwhelmed"
    stressed_multitask = "stressed_multitask"


class Role(str, Enum):
    client = "client"
    agent = "agent"


# ── Stage contracts ──────────────────────────────────────────


class DiceRollParams(BaseModel):
    seed: int
    complexity: Complexity
    sector: str
    topic: str
    outcome: Outcome
    style: Style
    client_archetype: ClientArchetype
    agent_archetype: AgentArchetype
    twist: str  # "none" or twist name, never None
    conflict_type: str
    emotional_arc: str
    n_messages: int  # total number of individual messages in dialogue
    use_dolphin: bool


class CharacterProfile(BaseModel):
    name: str
    mood: str
    personality: list[str]
    backstory: str = ""
    quirks: list[str] = []


class Characters(BaseModel):
    client: CharacterProfile
    agent: CharacterProfile


class MessagePlan(BaseModel):
    role: Role
    scene_id: int


class Scene(BaseModel):
    id: int
    beat: str  # "opening", "conflict", "twist", etc.
    description: str  # director note
    client_goal: str
    agent_goal: str
    emotional_state: str  # "client=frustrated(4), agent=steady(2)"
    expected_messages: list[str]  # ["client","agent"] etc.


class ScenePlan(BaseModel):
    scenes: list[Scene]
    total_messages: int


class Message(BaseModel):
    role: Role
    turn: int
    content: str
    emotion: Optional[str] = None
    intensity: Optional[int] = None  # 1-5, client only

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> str:
        """Accept Role enum, 'customer', 'user', 'caller' -> 'client';
        'support', 'assistant', 'bot', 'rep' -> 'agent'.
        Also handles Python 3.11+ str(Role.client)='Role.client' via .value."""
        # Extract the raw string value from Enum if needed
        raw = v.value if hasattr(v, "value") else v
        s = str(raw).lower().strip()
        if s in ("client", "agent"):
            return s
        client_aliases = ("customer", "user", "caller", "shopper", "buyer")
        agent_aliases = (
            "support",
            "assistant",
            "bot",
            "representative",
            "rep",
            "operator",
            "advisor",
            "staff",
            "employee",
            "service",
            "agent_",
        )
        if any(s.startswith(a) for a in client_aliases):
            return "client"
        if any(s.startswith(a) for a in agent_aliases):
            return "agent"
        # last-resort prefix match
        return "client" if s[0:3] in ("cli", "cus", "use") else "agent"

    @field_validator("turn", mode="before")
    @classmethod
    def coerce_turn(cls, v: object) -> int:
        """Accept '1', 1.0, etc."""
        try:
            return int(float(str(v).strip()))
        except (ValueError, TypeError):
            return 1


class EmotionTurn(BaseModel):
    client_emotion: str
    client_intensity: int  # 1-5
    agent_composure: Literal["steady", "holding", "slipping", "lost"]
    agent_stress: int  # 1-5

    @field_validator("client_emotion", mode="before")
    @classmethod
    def coerce_client_emotion(cls, v: object) -> str:
        if v is None or str(v).strip() in ("", "null", "none"):
            return "neutral"
        return str(v).strip()

    @field_validator("agent_composure", mode="before")
    @classmethod
    def coerce_composure(cls, v: object) -> str:
        """Normalise free-text LLM values to one of the four valid literals."""
        valid = ("steady", "holding", "slipping", "lost")
        if v in valid:
            return v  # type: ignore[return-value]
        v_low = str(v).lower().strip()
        if v_low in valid:
            return v_low
        # Prefix match (e.g. 'steadying' -> 'steady', 'slipp...' -> 'slipping')
        for val in valid:
            if v_low.startswith(val[:4]):
                return val
        # Keyword fallback
        _map = {
            "stable": "steady",
            "calm": "steady",
            "composed": "steady",
            "confident": "steady",
            "relaxed": "steady",
            "normal": "steady",
            "nervous": "holding",
            "anxious": "holding",
            "tense": "holding",
            "worried": "holding",
            "uncertain": "holding",
            "cautious": "holding",
            "stressed": "slipping",
            "struggling": "slipping",
            "frustrated": "slipping",
            "overwhelm": "slipping",
            "strain": "slipping",
            "difficult": "slipping",
            "panic": "lost",
            "broke": "lost",
            "broken": "lost",
            "collaps": "lost",
        }
        for key, mapped in _map.items():
            if key in v_low:
                return mapped
        return "holding"  # safe default

    @field_validator("client_intensity", "agent_stress", mode="before")
    @classmethod
    def coerce_int_1_5(cls, v: object) -> int:
        """Accept '3', 3.0, '3/5', etc. and clamp to 1-5."""
        try:
            n = int(float(str(v).split("/")[0].strip()))
        except (ValueError, TypeError):
            n = 3
        return max(1, min(5, n))


class EmotionAnalysis(BaseModel):
    turns: dict[str, EmotionTurn]  # key = str(turn_number)


class Dialogue(BaseModel):
    messages: list[Message]


class DirtMeta(BaseModel):
    applied: list[str] = []  # ["keyboard_typo:turn1", ...]


class DialogueOutput(BaseModel):
    id: str
    seed: int
    params: DiceRollParams
    characters: Characters
    messages: list[Message]
    meta: dict = {}


# ── Analysis contracts (for analyze.py) ─────────────────────


class AgentMistake(str, Enum):
    ignored_question = "ignored_question"
    incorrect_info = "incorrect_info"
    rude_tone = "rude_tone"
    no_resolution = "no_resolution"
    unnecessary_escalation = "unnecessary_escalation"
    premature_admission = "premature_admission"
    identity_not_verified = "identity_not_verified"


class Intent(str, Enum):
    payment_issue = "payment_issue"
    technical_error = "technical_error"
    account_access = "account_access"
    billing_question = "billing_question"
    refund = "refund"
    other = "other"


class Satisfaction(str, Enum):
    satisfied = "satisfied"
    neutral = "neutral"
    unsatisfied = "unsatisfied"


class Resolution(str, Enum):
    resolved = "resolved"
    partial = "partial"
    unresolved = "unresolved"


class AnalysisResult(BaseModel):
    dialogue_id: str
    intent: Intent
    satisfaction: Satisfaction
    quality_score: int  # 1-5
    agent_mistakes: list[AgentMistake] = []
    hidden_dissatisfaction: bool
    resolution: Resolution
    summary: str
