"""
FewShotBank — loads bitext_instructions.jsonl and groups examples by mood/style.
Provides .bank: dict[str, list[str]] for use in stage4_client_style.
"""

import json
from collections import defaultdict
from pathlib import Path

# Map bitext moods → our style names
_MOOD_MAP = {
    "aggressive": "aggressive",
    "casual": "casual",
    "formal": "formal",
    "passive_aggressive": "passive_aggressive",
    # bitext uses short codes; catch extras
    "polite": "formal",
    "upset": "aggressive",
    "anxious": "casual",
}

_MAX_EXAMPLES = 50  # cap per style to keep memory sane


class FewShotBank:
    def __init__(self, path: str = "corpus/bitext_instructions.jsonl"):
        self.bank: dict[str, list[str]] = defaultdict(list)
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_mood = obj.get("mood", "").lower().strip()
                style = _MOOD_MAP.get(raw_mood, "casual")
                if len(self.bank[style]) < _MAX_EXAMPLES:
                    self.bank[style].append(obj.get("instruction", ""))

        # Ensure all four canonical styles exist
        for style in ("casual", "aggressive", "formal", "passive_aggressive"):
            if style not in self.bank:
                self.bank[style] = []
