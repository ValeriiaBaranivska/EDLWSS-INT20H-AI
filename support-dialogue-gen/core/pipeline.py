"""
Pipeline orchestrator — runs all stages and returns a DialogueOutput.
"""

import time
import uuid

from config import MODELS
from pipeline import (
    stage0_dice,
    stage1_characters,
    stage2_story,
    stage6_dirt,
)
from pipeline.few_shot_bank import FewShotBank

from core.llm_client import LLMClient
from core.models import (
    CharacterProfile,
    Characters,
    DialogueOutput,
    Message,
    Role,
)

# ── bot-phrase cleaner ────────────────────────────────────────
_BOT_STARTS = [
    "i apologize for",
    "i'm sorry to hear",
    "i'd be happy to",
    "certainly!",
    "of course!",
    "thank you for your patience",
    "i sincerely apologize",
    "i completely understand your frustration",
    "i know how frustrating",
    "i understand how frustrating",
    "i can see that",
    "that must be",
    "we'll get this sorted",
    "i completely understand",
    "i understand your frustration",
    "i apologize for any",
    "let me help you with",
    "i would be happy",
    "absolutely!",
    "no problem!",
]


def strip_bot_phrases(messages: list[Message]) -> list[Message]:
    result = []
    for msg in messages:
        if msg.role == Role.agent:
            content = msg.content
            for phrase in _BOT_STARTS:
                if content.lower().startswith(phrase):
                    rest = content[len(phrase) :].lstrip(", .")
                    if len(rest) > 10:
                        content = rest[0].upper() + rest[1:]
                    break
            result.append(msg.model_copy(update={"content": content}))
        else:
            result.append(msg)
    return result


# ── empty-messages guard ──────────────────────────────────────
def guard_empty(messages: list[Message], stage_name: str) -> list[Message]:
    if not messages:
        raise ValueError(f"Stage {stage_name} returned 0 messages")
    if len(messages) < 2:
        raise ValueError(f"Stage {stage_name} returned < 2 messages: {len(messages)}")
    return messages


class Pipeline:
    def __init__(self):
        self.client = LLMClient()
        self.few_shots = FewShotBank("corpus/bitext_instructions.jsonl")

    def run(self, seed: int | None = None) -> DialogueOutput:

        t0 = time.time()
        stages_run: list[str] = []
        stages_skipped: list[str] = []
        dirt_applied: list[str] = []

        # ── Stage 0: dice roll ────────────────────────────────
        params = stage0_dice.run(seed)
        stages_run.append("0")

        # ── ALWAYS use simple path with qwen (no llama voice) ──
        # Generate characters first for persona context
        characters = stage1_characters.run(self.client, params)
        stages_run.append("1")

        # Use stage2 for better story generation (like old batch_20)
        messages = stage2_story.run(self.client, params, characters)
        messages = guard_empty(messages, "2")
        messages = strip_bot_phrases(messages)
        stages_run.append("2")
        stages_skipped.extend(["2a", "2b", "3", "4", "4B", "5"])

        # ── Stage 6: dirt ─────────────────────────────────────
        messages, dirt_applied = stage6_dirt.DirtLayer(
            slang_bank_path="corpus/slang_bank.json",
            filler_bank_path="corpus/filler_bank.json",
            seed=params.seed,
        ).apply(messages, params)
        stages_run.append("6")

        return DialogueOutput(
            id=str(uuid.uuid4()),
            seed=params.seed,
            params=params,
            characters=characters,
            messages=messages,
            meta={
                "total_time_s": round(time.time() - t0, 2),
                "model_used": MODELS["dolphin"]
                if params.use_dolphin
                else MODELS["writer"],
                "stages_run": stages_run,
                "stages_skipped": stages_skipped,
                "dirt_applied": dirt_applied,
            },
        )


def _default_characters() -> Characters:
    return Characters(
        client=CharacterProfile(
            name="Customer",
            mood="neutral",
            personality=["casual"],
            backstory="",
        ),
        agent=CharacterProfile(
            name="Agent",
            mood="neutral",
            personality=["professional"],
            quirks=[],
        ),
    )
