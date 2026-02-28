import random

from core.models import (
    AgentArchetype,
    ClientArchetype,
    DiceRollParams,
)

SECTORS = [
    "retail",
    "telco",
    "banking",
    "travel",
    "healthcare",
    "insurance",
    "hospitality",
]

TOPICS = [
    "order stuck in transit",
    "billing error",
    "wrong item received",
    "cancellation request denied",
    "subscription payment failed",
    "overdraft fee dispute",
    "suspicious transaction",
    "appointment scheduling issue",
    "service outage",
    "account locked",
    "refund not received",
    "price dispute",
    "delivery to wrong address",
    "plan upgrade issue",
    "loyalty points missing",
    "double charge",
    "contract termination fee",
    "technical support failure",
]

TWISTS = {
    "none": 40,
    "client_calms_down": 6,
    "client_realizes_own_fault": 6,
    "agent_breaks_protocol": 6,
    "agent_makes_mistake": 6,
    "company_acknowledges_fault": 5,
    "unresolved_escalate": 5,
    "ragequit": 5,
    "personal_crisis_revealed": 5,
    "legal_escalation": 5,
    "resolution_after_years": 3,
    "wrong_department_entirely": 3,
}

OUTCOME_WEIGHTS = {
    "resolved_quick": 20,
    "resolved_neutral": 20,
    "unresolved_passive": 15,
    "unresolved_ragequit": 15,
    "conflict": 15,
    "info_only": 15,
}

STYLE_WEIGHTS = {
    "casual": 35,
    "aggressive": 30,
    "formal": 15,
    "passive_aggressive": 20,
}

USE_DOLPHIN_ARCHETYPES = ["burned_out", "hands_tied"]
USE_DOLPHIN_TWISTS = [
    "agent_breaks_protocol",
    "ragequit",
    "legal_escalation",
    "unresolved_escalate",
]


def run(seed: int | None = None) -> DiceRollParams:
    if seed is None:
        seed = random.randint(0, 999_999)
    rng = random.Random(seed)

    complexity = rng.choices(["simple", "complex"], weights=[40, 60])[0]

    # n_messages = total individual messages (not turns)
    # Обмежено до 2-8 повідомлень
    n_messages = rng.randint(2, 5) if complexity == "simple" else rng.randint(4, 8)

    twist = rng.choices(
        list(TWISTS.keys()),
        weights=list(TWISTS.values()),
    )[0]

    agent_archetype = rng.choice(list(AgentArchetype)).value
    client_archetype = rng.choice(list(ClientArchetype)).value

    use_dolphin = (
        agent_archetype in USE_DOLPHIN_ARCHETYPES or twist in USE_DOLPHIN_TWISTS
    )

    return DiceRollParams(
        seed=seed,
        complexity=complexity,
        sector=rng.choice(SECTORS),
        topic=rng.choice(TOPICS),
        outcome=rng.choices(
            list(OUTCOME_WEIGHTS.keys()),
            weights=list(OUTCOME_WEIGHTS.values()),
        )[0],
        style=rng.choices(
            list(STYLE_WEIGHTS.keys()),
            weights=list(STYLE_WEIGHTS.values()),
        )[0],
        client_archetype=client_archetype,
        agent_archetype=agent_archetype,
        twist=twist,
        conflict_type=rng.choice(
            [
                "policy_dispute",
                "billing_error",
                "technical_issue",
                "delivery_problem",
                "account_access",
                "wrong_info_given",
                "client_own_fault",
                "company_fault",
                "none",
            ]
        ),
        emotional_arc=rng.choice(
            [
                "escalates",
                "deescalates",
                "stable",
                "rollercoaster",
            ]
        ),
        n_messages=n_messages,
        use_dolphin=use_dolphin,
    )
