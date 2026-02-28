import json
import os
import re
from typing import Optional

import requests


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] or } (common LLM mistake)."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def _fix_truncated_json(text: str) -> str:
    """Attempt to close truncated JSON by adding missing brackets."""
    # Count open/close brackets
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    # Remove any trailing incomplete key-value pairs
    # e.g., '"key": ' or '"key": "incomplete'
    text = re.sub(r',?\s*"[^"]*":\s*("?[^,}\]]*)?$', "", text)
    text = re.sub(r',?\s*"[^"]*$', "", text)  # incomplete key

    # Close open structures
    text = text.rstrip(", \n\t")
    text += "}" * max(0, open_braces)
    text += "]" * max(0, open_brackets)

    return text


def _extract_all_json_objects(text: str) -> list[dict]:
    """Extract all valid JSON objects from text, handling nested braces."""
    objects = []
    depth = 0
    start = None

    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(_strip_trailing_commas(text[start : i + 1]))
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    return objects


def safe_json_parse(text: str) -> dict:
    """Strip markdown fences, fix trailing commas, parse JSON.
    Falls back to extracting the first [...] or {...} block.
    Also handles multiple JSON objects that need merging (e.g., separate scene blocks)."""
    text = re.sub(r"```json|```", "", text).strip()

    # Try as-is first
    for candidate in [text, _strip_trailing_commas(text)]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try fixing truncated JSON
    for candidate in [
        _fix_truncated_json(text),
        _strip_trailing_commas(_fix_truncated_json(text)),
    ]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try to extract the first JSON array or object block
    for pattern in [r"\[.*\]", r"\{.*\}"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            for candidate in [
                match.group(),
                _strip_trailing_commas(match.group()),
                _fix_truncated_json(match.group()),
            ]:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Handle multiple JSON objects (e.g., llama outputting scene blocks separately)
    all_objects = _extract_all_json_objects(text)
    all_scenes = []
    for obj in all_objects:
        if "scenes" in obj and isinstance(obj["scenes"], list):
            all_scenes.extend(obj["scenes"])

    if all_scenes:
        return {"scenes": all_scenes}

    raise ValueError(f"Cannot parse JSON from: {text[:200]}")


class LLMClient:
    def __init__(self):
        self.backend = os.getenv("LLM_BACKEND", "ollama")
        # ollama | openai | anthropic | huggingface

        # Ollama
        self.ollama_url = os.getenv(
            "OLLAMA_URL", "http://localhost:11434/v1/chat/completions"
        )

        # OpenAI-compatible API
        self.api_url = os.getenv("API_URL", "")
        self.api_key = os.getenv("API_KEY", "")

        # HuggingFace Inference API
        self.hf_api_key = os.getenv("HF_API_KEY", "")
        self._hf_client = None  # lazy init

        # Model overrides from env
        self.model_default = os.getenv("MODEL_DEFAULT", "qwen2.5:7b")
        self.model_writer = os.getenv("MODEL_WRITER", "llama3.1:8b")
        self.model_parser = os.getenv("MODEL_PARSER", "qwen2.5:7b")
        self.model_dolphin = os.getenv("MODEL_DOLPHIN", "dolphin-mistral")

    def _get_hf_client(self):
        if self._hf_client is None:
            from huggingface_hub import InferenceClient

            self._hf_client = InferenceClient(api_key=self.hf_api_key)
        return self._hf_client

    def complete(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        backend: Optional[str] = None,  # overrides instance-level backend
    ) -> str:

        model = model or self.model_default
        effective = backend or self.backend

        if effective == "ollama":
            return self._ollama(system, user, model, temperature, max_tokens)
        elif effective in ["openai", "anthropic"]:
            return self._api(system, user, model, temperature, max_tokens)
        elif effective == "huggingface":
            return self._huggingface(system, user, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown backend: {effective}")

    def _ollama(self, system, user, model, temperature, max_tokens) -> str:
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = requests.post(self.ollama_url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _api(self, system, user, model, temperature, max_tokens) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _huggingface(self, system, user, model, temperature, max_tokens) -> str:
        hf = self._get_hf_client()
        response = hf.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
