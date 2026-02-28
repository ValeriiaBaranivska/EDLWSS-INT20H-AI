"""
Stage 6 — Dirt Layer (v2)

Injects realistic noise into dialogues using archetype-aware DIRT_PROFILES.

Client message transforms (controlled by profile per archetype+intensity):
  typos:     keyboard_typo, transposition, missing_letter, typo_rules, wrong_spaces
  register:  lowercase, caps_burst (with arc-position suppression)
  discourse: filler_prepend, slang_sub (style-aware with usage limits), ellipsis
  structure: punct_drop, split (disabled for dolphin messages)

Agent message transforms (archetype-based, NEVER adds typos):
  burned_out          → drop_last_sentence (35%)
  newbie_overwhelmed  → hesitation_prepend (40%)
  stressed_multitask  → multitask_interrupt (30%)

Dialogue-level events (checked once in apply() before the message loop):
  wrong_keyboard_event: 4% per dialogue — garbles first consecutive client block

Split handling:
  If _dirty_client() produces a split, it returns the Message with content set
  to a list[str].  apply() detects this and inserts each part as a separate
  Message object (same turn, same role).

Returns (list[Message], list[str]) where list[str] records applied transforms.
"""

import json
import random
import re
from typing import Optional

from core.models import DiceRollParams, Message, Role

# ── KEYBOARD NEIGHBORS (QWERTY proximity) ─────────────────────────────────────

KEYBOARD_NEIGHBORS: dict[str, list[str]] = {
    "a": ["s", "q", "w"],
    "b": ["v", "g", "h", "n"],
    "c": ["x", "d", "f", "v"],
    "d": ["s", "e", "r", "f", "c", "x"],
    "e": ["w", "r", "d", "s"],
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "y", "h", "b", "v"],
    "h": ["g", "y", "u", "j", "n", "b"],
    "i": ["u", "o", "k", "j"],
    "j": ["h", "u", "i", "k", "m", "n"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "m": ["n", "j", "k"],
    "n": ["b", "h", "j", "m"],
    "o": ["i", "p", "l", "k"],
    "p": ["o", "l"],
    "r": ["e", "t", "f", "d"],
    "s": ["a", "w", "e", "d", "x", "z"],
    "t": ["r", "y", "g", "f"],
    "u": ["y", "i", "j", "h"],
    "v": ["c", "f", "g", "b"],
    "w": ["q", "e", "s", "a"],
    "x": ["z", "s", "d", "c"],
    "y": ["t", "u", "h", "g"],
    "z": ["a", "s", "x"],
}

# ── TYPO RULES (common word-level transposition mistakes) ─────────────────────

TYPO_RULES: dict[str, list[str]] = {
    "the": ["teh", "hte"],
    "have": ["ahve", "hvae"],
    "that": ["taht", "htat"],
    "your": ["yuor", "yoru"],
    "with": ["wiht", "iwth"],
    "just": ["jsut", "ujst"],
    "what": ["waht", "hwat"],
    "when": ["wehn", "whn"],
    "they": ["tehy", "thye"],
    "been": ["bene", "bnee"],
    "this": ["tihs", "thsi"],
    "from": ["form", "fomr"],
    "about": ["aobut", "abotu"],
    "charged": ["chraged", "charegd"],
    "payment": ["paymnet", "pyament"],
    "refund": ["refudn", "rfund"],
    "account": ["acocunt", "accoutn"],
    "problem": ["problme", "probelm"],
    "working": ["workign", "wrking"],
}

# ── CAPS words always kept uppercase during suppression ───────────────────────

KEEP_CAPS: set[str] = {"WTF", "BS", "ASAP", "FYI", "OK", "OMG"}

# ── QWERTY → Ukrainian layout mapping ─────────────────────────────────────────

QWERTY_TO_UA: dict[str, str] = {
    "q": "й",
    "w": "ц",
    "e": "у",
    "r": "к",
    "t": "е",
    "y": "н",
    "u": "г",
    "i": "ш",
    "o": "щ",
    "p": "з",
    "a": "ф",
    "s": "і",
    "d": "в",
    "f": "а",
    "g": "п",
    "h": "р",
    "j": "о",
    "k": "л",
    "l": "д",
    "z": "я",
    "x": "ч",
    "c": "с",
    "v": "м",
    "b": "и",
    "n": "т",
    "m": "ь",
    " ": " ",
}

LAYOUT_APOLOGIES: list[str] = [
    "sorry wrong keyboard lol",
    "oops wrong layout",
    "*wrong keyboard",
    "sorry, meant to type in english",
    "lol wrong language",
]

# ── ELLIPSIS replacements by style ────────────────────────────────────────────

ELLIPSIS_REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "passive_aggressive": [
        ("okay", "okay..."),
        ("sure", "sure..."),
        ("fine", "fine..."),
        ("thank you", "thank you..."),
        ("I see", "I see..."),
        ("I understand", "I understand..."),
        ("noted", "noted..."),
        ("alright", "alright..."),
        ("seriously?", "seriously..?"),
        ("right?", "right..?"),
    ],
    "hidden_dissatisfaction": [
        ("alright", "alright..."),
        ("understood", "understood..."),
        ("I'll wait", "I'll wait..."),
        ("I'll check", "I'll check..."),
        ("okay", "okay..."),
        ("thanks", "thanks..."),
        ("I guess", "I guess..."),
    ],
}

# ── SLANG substitutions by style ──────────────────────────────────────────────

SLANG_BY_STYLE: dict[str, dict[str, str]] = {
    "aggressive": {
        "What the heck": "wtf",
        "What the hell": "wtf",
        "This is nonsense": "this is bs",
        "as soon as possible": "asap",
        "seriously": "srsly",
        "please": "pls",
        "to be honest": "tbh",
    },
    "casual": {
        "laughing out loud": "lol",
        "oh my god": "omg",
        "to be honest": "tbh",
        "as soon as possible": "asap",
        "please": "pls",
        "because": "bc",
        "though": "tho",
        "I don't know": "idk",
        "by the way": "btw",
    },
    "passive_aggressive": {
        "to be honest": "tbh",
        "by the way": "btw",
        "I suppose": "ig",
    },
    "formal": {},
}

SLANG_MAX_USES: dict[str, int] = {
    "lol": 2,
    "omg": 2,
    "tbh": 1,
    "wtf": 3,
    "asap": 2,
    "pls": 2,
    "bc": 2,
    "tho": 2,
    "idk": 2,
    "btw": 2,
}

# ── CLEAN OUTPUT — meta-leak strings ──────────────────────────────────────────

_META_LEAKS: list[str] = [
    "next message:",
    "last message:",
    "(note:",
    "rewritten version:",
    "here is the",
    "remember to keep",
    "absolute rules",
    "as the customer",
    "i'll rewrite",
    "rewritten:",
    "here's the rewritten",
    "here is the rewritten",
    "rewritten message:",
    "rewrite this",
    "rewrite this customer",
    "in your voice",
    "as a representative of our company",
    "on behalf of our company",
    "{{website_url}}",
    "visit our website at {{",
    "representative of",
]

# ── WRONG KEYBOARD EVENT probability (dialogue-level) ─────────────────────────

WRONG_KB_PROBABILITY: float = 0.04

# ── DIRT PROFILES (archetype + intensity bucket) ──────────────────────────────

DIRT_PROFILES: dict[tuple[str, str], dict] = {
    ("karen", "low"): {
        "typo_types": ["transposition"],
        "typo_rate": 0.08,
        "lowercase_rate": 0.00,
        "caps_rate": 0.15,
        "filler_rate": 0.05,
        "slang_rate": 0.00,
        "punct_drop_rate": 0.20,
        "split_rate": 0.05,
        "ellipsis_rate": 0.00,
    },
    ("karen", "high"): {
        "typo_types": ["transposition", "keyboard_typo"],
        "typo_rate": 0.25,
        "lowercase_rate": 0.00,
        "caps_rate": 0.70,  # SCREAMING
        "filler_rate": 0.00,
        "slang_rate": 0.10,  # wtf, bs
        "punct_drop_rate": 0.10,
        "split_rate": 0.15,
        "ellipsis_rate": 0.00,
    },
    ("angry_veteran", "low"): {
        "typo_types": ["transposition"],
        "typo_rate": 0.10,
        "lowercase_rate": 0.00,
        "caps_rate": 0.20,
        "filler_rate": 0.05,
        "slang_rate": 0.05,
        "punct_drop_rate": 0.20,
        "split_rate": 0.08,
        "ellipsis_rate": 0.00,
    },
    ("angry_veteran", "high"): {
        "typo_types": ["transposition", "keyboard_typo"],
        "typo_rate": 0.30,
        "lowercase_rate": 0.00,
        "caps_rate": 0.55,
        "filler_rate": 0.00,
        "slang_rate": 0.15,
        "punct_drop_rate": 0.15,
        "split_rate": 0.12,
        "ellipsis_rate": 0.00,
    },
    ("elderly_confused", "low"): {
        "typo_types": ["missing_letter", "typo_rules"],
        "typo_rate": 0.40,  # types slowly, makes word errors
        "lowercase_rate": 0.60,  # doesn't know caps
        "caps_rate": 0.00,
        "filler_rate": 0.30,  # "um", "well", "so"
        "slang_rate": 0.00,  # no slang ever
        "punct_drop_rate": 0.50,
        "split_rate": 0.05,
        "ellipsis_rate": 0.05,
    },
    ("elderly_confused", "high"): {
        "typo_types": ["missing_letter", "typo_rules"],
        "typo_rate": 0.55,
        "lowercase_rate": 0.70,
        "caps_rate": 0.00,
        "filler_rate": 0.40,
        "slang_rate": 0.00,
        "punct_drop_rate": 0.60,
        "split_rate": 0.08,
        "ellipsis_rate": 0.10,
    },
    ("tech_confused", "low"): {
        "typo_types": ["wrong_spaces", "missing_letter"],
        "typo_rate": 0.20,
        "lowercase_rate": 0.30,
        "caps_rate": 0.05,
        "filler_rate": 0.20,
        "slang_rate": 0.05,
        "punct_drop_rate": 0.30,
        "split_rate": 0.10,
        "ellipsis_rate": 0.05,
    },
    ("young_professional", "low"): {
        "typo_types": ["keyboard_typo"],  # types fast on phone
        "typo_rate": 0.15,
        "lowercase_rate": 0.50,  # phone typing
        "caps_rate": 0.05,
        "filler_rate": 0.15,
        "slang_rate": 0.40,  # tbh, ngl, lmk, idk
        "punct_drop_rate": 0.60,  # never uses periods
        "split_rate": 0.20,
        "ellipsis_rate": 0.00,
    },
    ("young_professional", "high"): {
        "typo_types": ["keyboard_typo", "transposition"],
        "typo_rate": 0.25,
        "lowercase_rate": 0.40,
        "caps_rate": 0.20,
        "filler_rate": 0.10,
        "slang_rate": 0.35,
        "punct_drop_rate": 0.55,
        "split_rate": 0.20,
        "ellipsis_rate": 0.00,
    },
    ("conspirologist", "low"): {
        "typo_types": ["keyboard_typo"],
        "typo_rate": 0.10,
        "lowercase_rate": 0.15,
        "caps_rate": 0.40,  # emphasizes key words
        "filler_rate": 0.05,
        "slang_rate": 0.05,
        "punct_drop_rate": 0.30,
        "split_rate": 0.10,
        "ellipsis_rate": 0.15,
    },
    ("conspirologist", "high"): {
        "typo_types": ["keyboard_typo"],
        "typo_rate": 0.15,
        "lowercase_rate": 0.10,
        "caps_rate": 0.60,
        "filler_rate": 0.00,
        "slang_rate": 0.05,
        "punct_drop_rate": 0.20,
        "split_rate": 0.15,
        "ellipsis_rate": 0.10,
    },
    ("grieving", "low"): {
        "typo_types": ["missing_letter"],
        "typo_rate": 0.15,
        "lowercase_rate": 0.40,
        "caps_rate": 0.00,
        "filler_rate": 0.25,
        "slang_rate": 0.05,
        "punct_drop_rate": 0.40,
        "split_rate": 0.15,
        "ellipsis_rate": 0.20,  # trailing off
    },
    ("calling_bluff", "high"): {
        "typo_types": ["transposition", "keyboard_typo"],
        "typo_rate": 0.20,
        "lowercase_rate": 0.00,
        "caps_rate": 0.50,
        "filler_rate": 0.05,
        "slang_rate": 0.15,
        "punct_drop_rate": 0.15,
        "split_rate": 0.12,
        "ellipsis_rate": 0.00,
    },
    ("passive_aggressive", "any"): {
        # style=passive_aggressive overrides archetype lookup
        "typo_types": [],
        "typo_rate": 0.05,
        "lowercase_rate": 0.10,
        "caps_rate": 0.00,
        "filler_rate": 0.10,
        "slang_rate": 0.20,
        "punct_drop_rate": 0.10,
        "split_rate": 0.08,
        "ellipsis_rate": 0.35,  # "okay...", "fine...", "I see..."
    },
}

DEFAULT_DIRT_PROFILE: dict = {
    "typo_types": ["keyboard_typo"],
    "typo_rate": 0.12,
    "lowercase_rate": 0.20,
    "caps_rate": 0.10,
    "filler_rate": 0.15,
    "slang_rate": 0.15,
    "punct_drop_rate": 0.25,
    "split_rate": 0.08,
    "ellipsis_rate": 0.05,
}


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILE LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════


def get_dirt_profile(archetype: str, intensity: int, style: str) -> dict:
    """
    Returns dirt profile for this message.

    Priority:
      1. style == passive_aggressive → use passive_aggressive profile
      2. exact (archetype, bucket) match
      3. (archetype, "low") fallback — mid bucket scales rates by 1.3x
      4. DEFAULT_DIRT_PROFILE
    """
    if style == "passive_aggressive":
        return DIRT_PROFILES.get(("passive_aggressive", "any"), DEFAULT_DIRT_PROFILE)

    bucket = "low" if intensity <= 2 else "high" if intensity >= 4 else "mid"

    key = (archetype, bucket)
    if key in DIRT_PROFILES:
        return DIRT_PROFILES[key]

    key_low = (archetype, "low")
    if key_low in DIRT_PROFILES:
        profile = dict(DIRT_PROFILES[key_low])
        if bucket == "mid":
            for k in ("typo_rate", "caps_rate", "slang_rate"):
                profile[k] = min(profile[k] * 1.3, 0.8)
        return profile

    return DEFAULT_DIRT_PROFILE


# ═══════════════════════════════════════════════════════════════════════════════
# TYPO HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _keyboard_typo(word: str, rng: random.Random) -> str:
    """Replace one random character with an adjacent QWERTY key (words ≥3)."""
    if len(word) < 3:
        return word
    idx = rng.randint(0, len(word) - 1)
    ch = word[idx].lower()
    if ch in KEYBOARD_NEIGHBORS:
        replacement = rng.choice(KEYBOARD_NEIGHBORS[ch])
        return word[:idx] + replacement + word[idx + 1 :]
    return word


def _apply_transposition(word: str, rng: random.Random) -> str:
    """Swap two adjacent characters in word ≥3 chars."""
    if len(word) < 3:
        return word
    idx = rng.randint(0, len(word) - 2)
    return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]


def _apply_missing_letter(word: str, rng: random.Random) -> str:
    """Drop one letter (not first, not last) from word ≥4 chars."""
    if len(word) < 4:
        return word
    idx = rng.randint(1, len(word) - 2)
    return word[:idx] + word[idx + 1 :]


def _apply_typo_rules(text: str, rng: random.Random) -> str:
    """Apply TYPO_RULES word substitutions (30% chance per matching word)."""
    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in TYPO_RULES and rng.random() < 0.30:
            typo = rng.choice(TYPO_RULES[lower])
            if word[0].isupper():
                typo = typo[0].upper() + typo[1:]
            result.append(typo)
        else:
            result.append(word)
    return " ".join(result)


def _apply_wrong_spaces(text: str, rng: random.Random) -> str:
    """Remove one space after comma or period (chosen randomly among candidates)."""
    matches = list(re.finditer(r"[,.](?= [a-zA-Z])", text))
    if not matches:
        return text
    match = rng.choice(matches)
    return text[: match.end()] + text[match.end() + 1 :]


# ═══════════════════════════════════════════════════════════════════════════════
# CAPS SUPPRESSION
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_caps_suppression(text: str, arc_pos: float, rng: random.Random) -> str:
    """
    Suppress CAPS based on arc position (0.0 = start, 1.0 = end).

    Early (<0.33):  lowercase all words except KEEP_CAPS set.
    Mid (0.33-0.66): randomly lowercase 40% of words.
    Late (>0.66):   full caps allowed — return as-is.
    """
    if arc_pos >= 0.66:
        return text
    words = text.split()
    result = []
    for word in words:
        bare = word.strip(".,!?;:")
        if bare in KEEP_CAPS:
            result.append(word)
        elif arc_pos < 0.33:
            result.append(word.lower())
        else:
            result.append(word.lower() if rng.random() < 0.40 else word)
    return " ".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_split(text: str, rng: random.Random) -> list[str]:
    """
    Split message at natural break point.

    Priority 1: sentence boundary (.!? followed by space).
    Priority 2: clause boundary (comma + space, both sides ≥5 words).

    Returns [text] if no suitable break found, else [part1, part2].
    Capitalization of part2 inherits from part1 (lower → lower).
    """
    sentence_breaks = [m.end() for m in re.finditer(r"[.!?]\s+", text)]

    clause_breaks = []
    for m in re.finditer(r",\s+", text):
        before_words = len(text[: m.start()].split())
        after_words = len(text[m.end() :].split())
        if before_words >= 5 and after_words >= 5:
            clause_breaks.append(m.end())

    candidates = sentence_breaks or clause_breaks
    if not candidates:
        return [text]

    split_point = rng.choice(candidates)
    part1 = text[:split_point].rstrip()
    part2 = text[split_point:].strip()

    if not part1 or not part2:
        return [text]

    # Inherit capitalization from part1
    if part1 and part1[0].islower() and part2:
        part2 = part2[0].lower() + part2[1:] if len(part2) > 1 else part2.lower()

    return [part1, part2]


# ═══════════════════════════════════════════════════════════════════════════════
# ELLIPSIS
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_ellipsis(text: str, style: str) -> str:
    """Apply ellipsis substitution for passive_aggressive/hidden_dissatisfaction."""
    for original, replacement in ELLIPSIS_REPLACEMENTS.get(style, []):
        idx = text.lower().find(original.lower())
        if idx != -1:
            return text[:idx] + replacement + text[idx + len(original) :]
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT SNIPPET
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_layout_snippet(text: str, rng: random.Random) -> str:
    """
    Convert first 2-4 words through QWERTY_TO_UA mapping.
    70% chance: append a layout apology as a new line.
    """
    words = text.split()
    if not words:
        return text
    max_idx = min(len(words) - 1, 4)
    slice_len = rng.randint(2, max(2, max_idx))
    garbled_words = [
        "".join(QWERTY_TO_UA.get(ch.lower(), ch) for ch in word)
        for word in words[:slice_len]
    ]
    rest = " ".join(words[slice_len:])
    result = " ".join(garbled_words) + (" " + rest if rest else "")
    if rng.random() < 0.70:
        result = result + "\n" + rng.choice(LAYOUT_APOLOGIES)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# WRONG KEYBOARD EVENT (dialogue-level)
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_wrong_keyboard_event(
    messages: list[Message],
    rng: random.Random,
) -> tuple[list[Message], list[str]]:
    """
    Find first consecutive client block (messages[i] and messages[i+1] both client).
    Garble first 2-4 words of messages[i] through QWERTY_TO_UA.
    Replace messages[i+1].content with a layout apology.
    """
    event_idx: Optional[int] = None
    for i in range(len(messages) - 1):
        if messages[i].role == Role.client and messages[i + 1].role == Role.client:
            event_idx = i
            break

    if event_idx is None:
        return messages, []

    result = list(messages)

    # Garble first 2-4 words of the first message in the consecutive block
    words = result[event_idx].content.split()
    num_words = 3 if rng.random() < 0.70 else 2
    num_words = min(num_words, len(words))
    garbled_words = [
        "".join(QWERTY_TO_UA.get(ch.lower(), ch) for ch in word)
        for word in words[:num_words]
    ]
    new_content = " ".join(garbled_words + words[num_words:])
    result[event_idx] = result[event_idx].model_copy(update={"content": new_content})

    # Replace the follow-up message with a layout apology
    apology = rng.choice(LAYOUT_APOLOGIES)
    result[event_idx + 1] = result[event_idx + 1].model_copy(
        update={"content": apology}
    )

    return result, ["wrong_keyboard_event"]


# ═══════════════════════════════════════════════════════════════════════════════
# CLEAN OUTPUT VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════


def _clean_output(text: str, fallback: str) -> str:
    """
    Validate transformed text; return fallback on failure.

    Fails if: empty, wrapped in quotes with meta content, meta-leak phrase
    detected, or text is suspiciously short and ends mid-sentence.
    """
    if not text:
        return fallback

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    text_lower = text.lower()
    if any(leak in text_lower for leak in _META_LEAKS):
        return fallback

    ends_naturally = bool(text) and text[-1] in ".!?)"
    too_short = len(text.split()) < len(fallback.split()) * 0.5
    if not ends_naturally and too_short:
        return fallback

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# PRIORITY SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


def _select_by_priority(candidates: list[tuple], max_n: int) -> list[tuple]:
    """Sort candidates by TRANSFORM_PRIORITY order, return top max_n."""
    _PRIORITY = [
        "keyboard_typo",
        "wrong_layout",
        "ellipsis",
        "filler_sound",
        "laugh_typo",
        "wrong_spaces",
        "brackets",
        "slang",
        "emoji",
        "text_smiley",
        "hello_pattern",
        "split",
        # internal candidate names
        "typo_keyboard_typo",
        "typo_transposition",
        "typo_missing_letter",
        "typo_wrong_spaces",
        "typo_typo_rules",
        "lowercase",
        "caps_burst",
        "filler_prepend",
        "slang_sub",
        "punct_drop",
    ]

    def rank(item: tuple) -> int:
        try:
            return _PRIORITY.index(item[0])
        except ValueError:
            return 99

    return sorted(candidates, key=rank)[:max_n]


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORM DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════


def _apply_transform(
    text: str,
    transform_type: str,
    style: str,
    rng: random.Random,
    filled_pauses: list,
    discourse_marks: list,
    slang_bank: list,
    slang_usage: dict,
    turn: int,
    total_turns: int,
) -> tuple[str, bool]:
    """
    Route transform_type to the correct helper function.
    Returns (new_text, was_applied).
    """

    if transform_type == "keyboard_typo":
        words = text.split()
        new_words = [
            _keyboard_typo(w, rng) if rng.random() < 0.50 else w for w in words
        ]
        new_text = " ".join(new_words)
        return new_text, new_text != text

    if transform_type == "transposition":
        words = text.split()
        new_words = [
            _apply_transposition(w, rng) if rng.random() < 0.50 else w for w in words
        ]
        new_text = " ".join(new_words)
        return new_text, new_text != text

    if transform_type == "missing_letter":
        words = text.split()
        new_words = [
            _apply_missing_letter(w, rng) if rng.random() < 0.50 else w for w in words
        ]
        new_text = " ".join(new_words)
        return new_text, new_text != text

    if transform_type == "typo_rules":
        new_text = _apply_typo_rules(text, rng)
        return new_text, new_text != text

    if transform_type == "wrong_spaces":
        new_text = _apply_wrong_spaces(text, rng)
        return new_text, new_text != text

    if transform_type == "lowercase":
        return text.lower(), True

    if transform_type == "caps_burst":
        arc_pos = (turn - 1) / max(total_turns - 1, 1)
        words = text.split()
        if not words:
            return text, False
        idx = rng.randint(0, len(words) - 1)
        words[idx] = words[idx].upper()
        new_text = _apply_caps_suppression(" ".join(words), arc_pos, rng)
        return new_text, True

    if transform_type == "filler_prepend":
        if not filled_pauses:
            return text, False
        weights = [fp.get("prob", 1.0) for fp in filled_pauses]
        filler = rng.choices(filled_pauses, weights=weights)[0]["word"]
        if not text:
            return text, False
        new_text = f"{filler}, {text[0].lower()}{text[1:]}"
        return new_text, True

    if transform_type == "slang_sub":
        slang_map = SLANG_BY_STYLE.get(style, {})
        if slang_map:
            entries = list(slang_map.items())
            rng.shuffle(entries)
            for expansion, acronym in entries:
                uses = slang_usage.get(acronym, 0)
                if uses >= SLANG_MAX_USES.get(acronym.lower(), 99):
                    continue
                pattern = re.compile(re.escape(expansion), re.IGNORECASE)
                new_text = pattern.sub(acronym, text, count=1)
                if new_text != text:
                    slang_usage[acronym] = uses + 1
                    return new_text, True
        elif slang_bank:
            weights = [s.get("weight", 1.0) for s in slang_bank]
            entry = rng.choices(slang_bank, weights=weights)[0]
            acronym = entry["acronym"]
            uses = slang_usage.get(acronym, 0)
            if uses < SLANG_MAX_USES.get(acronym.lower(), 99):
                pattern = re.compile(re.escape(entry["expansion"]), re.IGNORECASE)
                new_text = pattern.sub(acronym, text, count=1)
                if new_text != text:
                    slang_usage[acronym] = uses + 1
                    return new_text, True
        return text, False

    if transform_type == "punct_drop":
        if rng.random() < 0.40:
            new_text = text.rstrip(".!?,;")
            return new_text, new_text != text
        return text, False

    if transform_type == "ellipsis":
        new_text = _apply_ellipsis(text, style)
        return new_text, new_text != text

    # "split" is handled externally by _dirty_client — should not reach here
    return text, False


# ═══════════════════════════════════════════════════════════════════════════════
# DIRT LAYER CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class DirtLayer:
    def __init__(
        self,
        slang_bank_path: str = "corpus/slang_bank.json",
        filler_bank_path: str = "corpus/filler_bank.json",
        seed: int = 0,
    ):
        self.rng = random.Random(seed)

        with open(slang_bank_path, encoding="utf-8") as f:
            self.slang: list = json.load(f)

        with open(filler_bank_path, encoding="utf-8") as f:
            raw_filler = json.load(f)
        self.filled_pauses: list = raw_filler.get("filled_pauses", [])
        self.discourse_marks: list = raw_filler.get("discourse_markers", [])

    # ── public entry point ────────────────────────────────────────────────────

    def apply(
        self,
        messages: list[Message],
        params: DiceRollParams,
    ) -> tuple[list[Message], list[str]]:
        """
        Apply dirt transforms to all messages.

        Step 1: dialogue-level wrong_keyboard_event (4% chance).
        Step 2: per-message transforms via profile lookup.
        Step 3: if a split occurred, insert both parts as separate Messages.

        Returns (final_messages, dirt_log).
        """
        result = list(messages)
        dirt_log: list[str] = []

        # 1. Wrong keyboard event — fires before message loop
        if self.rng.random() < WRONG_KB_PROBABILITY:
            result, wk_log = _apply_wrong_keyboard_event(result, self.rng)
            dirt_log.extend(wk_log)

        # 2. Per-message transforms
        total_turns = max((m.turn for m in result), default=1)
        slang_usage: dict[str, int] = {}

        final: list[Message] = []
        for msg in result:
            if msg.role == Role.client:
                msg, applied = self._dirty_client(msg, params, total_turns, slang_usage)
                # Split: _dirty_client sets content to list[str] when split fires
                if isinstance(msg.content, list):
                    for part in msg.content:
                        final.append(msg.model_copy(update={"content": part}))
                else:
                    final.append(msg)
            else:
                msg, applied = self._dirty_agent(msg, params)
                final.append(msg)
            dirt_log.extend(applied)

        # 3. Warn if nothing applied for inherently dirty styles
        if not dirt_log and params.style in ("casual", "aggressive"):
            print(f"  [WARN] dirt: 0 transforms, style={params.style}")

        return final, dirt_log

    # ── client ────────────────────────────────────────────────────────────────

    def _dirty_client(
        self,
        msg: Message,
        params: DiceRollParams,
        total_turns: int,
        slang_usage: dict,
    ) -> tuple[Message, list[str]]:
        """
        Apply profile-driven transforms to a single client message.

        When split fires, returns Message with content=list[str];
        apply() unwraps this into two separate Message objects.
        """
        profile = get_dirt_profile(
            params.client_archetype,
            getattr(msg, "intensity", None) or 3,
            params.style,
        )
        text = msg.content
        applied: list[str] = []
        r = self.rng
        turn_tag = f"turn{msg.turn}"
        is_dolphin = getattr(params, "use_dolphin", False)

        candidates: list[tuple[str, str]] = []

        # typo
        if r.random() < profile["typo_rate"] and profile["typo_types"]:
            typo_type = r.choice(profile["typo_types"])
            candidates.append((f"typo_{typo_type}", typo_type))

        # lowercase
        if r.random() < profile["lowercase_rate"]:
            candidates.append(("lowercase", "lowercase"))

        # caps_burst — suppressed in early turns (arc < 0.33)
        if r.random() < profile["caps_rate"]:
            arc_pos = (msg.turn - 1) / max(total_turns - 1, 1)
            if arc_pos >= 0.33:
                candidates.append(("caps_burst", "caps_burst"))

        # filler_prepend
        if r.random() < profile["filler_rate"] and self.filled_pauses:
            candidates.append(("filler_prepend", "filler_prepend"))

        # slang_sub
        if r.random() < profile["slang_rate"]:
            candidates.append(("slang_sub", "slang_sub"))

        # punct_drop
        if r.random() < profile["punct_drop_rate"]:
            candidates.append(("punct_drop", "punct_drop"))

        # split (disabled for dolphin)
        if not is_dolphin and r.random() < profile.get("split_rate", 0):
            candidates.append(("split", "split"))

        # ellipsis
        if r.random() < profile.get("ellipsis_rate", 0):
            candidates.append(("ellipsis", "ellipsis"))

        # Apply top-priority transforms, max 2
        chosen = _select_by_priority(candidates, max_n=2)

        split_parts: Optional[list[str]] = None
        for name, transform_type in chosen:
            if transform_type == "split":
                parts = _apply_split(text, r)
                if len(parts) == 2:
                    split_parts = parts
                    applied.append(f"split:{turn_tag}")
                continue
            new_text, ok = _apply_transform(
                text,
                transform_type,
                params.style,
                r,
                self.filled_pauses,
                self.discourse_marks,
                self.slang,
                slang_usage,
                msg.turn,
                total_turns,
            )
            if ok:
                text = new_text
                applied.append(f"{name}:{turn_tag}")

        # Clean output validator
        text = _clean_output(text, msg.content)

        if split_parts:
            split_parts[0] = _clean_output(split_parts[0], msg.content)
            split_parts[1] = _clean_output(split_parts[1], split_parts[1])
            # Return content as list; apply() handles insertion of both parts
            return msg.model_copy(update={"content": split_parts}), applied  # type: ignore[arg-type]

        return msg.model_copy(update={"content": text}), applied

    # ── agent ─────────────────────────────────────────────────────────────────

    def _dirty_agent(
        self,
        msg: Message,
        params: DiceRollParams,
    ) -> tuple[Message, list[str]]:
        """
        Apply archetype-based noise to agent messages.
        NEVER adds typos to agent messages.
        """
        text = msg.content
        applied: list[str] = []
        archetype = params.agent_archetype

        # burned_out → drop_last_sentence (35%)
        if archetype == "burned_out" and self.rng.random() < 0.35:
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            if len(sentences) > 1:
                text = " ".join(sentences[:-1])
                applied.append(f"drop_last_sentence:turn{msg.turn}")

        # newbie_overwhelmed → hesitation_prepend (40%)
        if archetype == "newbie_overwhelmed" and self.rng.random() < 0.40:
            pauses = ["Um, ", "Uh, ", "So, ", "Right, "]
            text = self.rng.choice(pauses) + text[0].lower() + text[1:]
            applied.append(f"hesitation_prepend:turn{msg.turn}")

        # stressed_multitask → multitask_interrupt (30%)
        if archetype == "stressed_multitask" and self.rng.random() < 0.30:
            interrupts = [
                "Quick update: ",
                "Just a sec — ",
                "Checking now, ",
                "On it — ",
            ]
            text = self.rng.choice(interrupts) + text[0].lower() + text[1:]
            applied.append(f"multitask_interrupt:turn{msg.turn}")

        return msg.model_copy(update={"content": text}), applied
