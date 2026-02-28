"""
Dialogue Analyzer — reads dialogues from TXT, sends to GLM-4.7 (Z.AI),
writes structured analysis to JSON + human-readable summary to TXT.
"""

import json
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# ── config ──────────────────────────────────────────────────────────────────
API_KEY = "09d814a3c5df4ca395957dcc5eebdc18.lEasoqyUtpdhgrir"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = "glm-4.7"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_MSG = """\
You are a dialogue analyst. You MUST respond with ONLY raw JSON — no markdown, \
no code fences, no explanation. Just the JSON object."""

ANALYSIS_PROMPT = """\
Analyze this customer support dialogue. Return ONLY a JSON object.

Required JSON structure (follow exactly):
{{
  "intent": "<one of: payment_issue, technical_error, account_access, billing_question, refund_request, other>",
  "satisfaction": "<one of: satisfied, neutral, unsatisfied>",
  "quality_score": <integer 1-5>,
  "agent_mistakes": [<list of strings from: ignored_question, incorrect_info, rude_tone, no_resolution, unnecessary_escalation, slow_response, unclear_communication, lack_of_empathy>],
  "hidden_dissatisfaction": <boolean - true if client formally thanks but problem is NOT resolved>,
  "summary": "<2-3 sentence summary of the dialogue>"
}}

IMPORTANT RULES:
- intent: Choose based on the MAIN topic of the conversation
- satisfaction:
  * "satisfied" = problem resolved AND client is genuinely happy
  * "neutral" = problem partially resolved OR client is indifferent
  * "unsatisfied" = problem not resolved OR client is clearly upset/frustrated
- quality_score: 1=terrible, 2=poor, 3=acceptable, 4=good, 5=excellent
- agent_mistakes: List ALL mistakes you detect. Empty array [] if none.
- hidden_dissatisfaction: TRUE if client says "thanks" or "fine" but issue wasn't actually solved

DIALOGUE (topic={topic}, outcome={outcome}, style={style}):
{conversation}"""


# ── parse TXT ───────────────────────────────────────────────────────────────

def parse_dialogues_txt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"--- Dialogue \(seed=(\d+)\) ---", text)
    dialogues = []
    for i in range(1, len(blocks), 2):
        seed = int(blocks[i])
        body = blocks[i + 1].strip()
        lines = body.splitlines()
        meta = {}
        messages = []
        turn = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Topic:"):
                meta["topic"] = line.split(":", 1)[1].strip()
            elif line.startswith("Outcome:"):
                meta["outcome"] = line.split(":", 1)[1].strip()
            elif line.startswith("Style:"):
                meta["style"] = line.split(":", 1)[1].strip()
            elif line.startswith("Client:") or line.startswith("Agent:"):
                role, content = line.split(":", 1)
                turn += 1
                messages.append({
                    "role": role.strip().lower(),
                    "turn": turn,
                    "content": content.strip(),
                })
        dialogues.append({
            "seed": seed,
            "topic": meta.get("topic", ""),
            "outcome": meta.get("outcome", ""),
            "style": meta.get("style", ""),
            "messages": messages,
        })
    return dialogues


# ── GLM-4.7 analysis ───────────────────────────────────────────────────────

def extract_json(raw: str) -> dict:
    """Try hard to extract JSON from model response."""
    raw = raw.strip()
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?\s*```\s*$", "", raw)
    raw = raw.strip()
    # try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # find first { ... last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start:end + 1])
    raise json.JSONDecodeError("no JSON found", raw, 0)


def analyze_dialogue(dlg: dict) -> dict | None:
    conversation = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in dlg["messages"]
    )
    prompt = ANALYSIS_PROMPT.format(
        topic=dlg["topic"],
        outcome=dlg["outcome"],
        style=dlg["style"],
        conversation=conversation,
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            raw = resp.choices[0].message.content
            return extract_json(raw)
        except json.JSONDecodeError:
            print(f"  [!] seed={dlg['seed']} attempt {attempt+1}: bad JSON, retrying")
            time.sleep(1)
        except Exception as e:
            print(f"  [!] seed={dlg['seed']} attempt {attempt+1}: {e}")
            time.sleep(2)
    return None


# ── output ──────────────────────────────────────────────────────────────────

def build_json_output(dialogues, analyses):
    records = []
    for dlg, analysis in zip(dialogues, analyses):
        record = {
            "seed": dlg["seed"],
            "messages": dlg["messages"],
            "intent": analysis.get("intent", "other") if analysis else "other",
            "satisfaction": analysis.get("satisfaction", "neutral") if analysis else "neutral",
            "quality_score": analysis.get("quality_score", 3) if analysis else 3,
            "agent_mistakes": analysis.get("agent_mistakes", []) if analysis else [],
            "hidden_dissatisfaction": analysis.get("hidden_dissatisfaction", False) if analysis else False,
            "summary": analysis.get("summary", "") if analysis else "",
        }
        records.append(record)
    return records


def build_summary_txt(dialogues, analyses):
    """Build combined summary: dialogue + analysis side by side."""
    lines = []
    lines.append("=" * 70)
    lines.append("CUSTOMER SUPPORT DIALOGUE ANALYSIS")
    lines.append(f"Analyzed by: {MODEL}")
    lines.append(f"Total dialogues: {len(dialogues)}")
    lines.append("=" * 70)

    for dlg, analysis in zip(dialogues, analyses):
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"DIALOGUE #{dlg['seed']}")
        lines.append("-" * 70)
        lines.append("")

        # Print conversation
        lines.append("CONVERSATION:")
        for m in dlg["messages"]:
            role = m["role"].upper()
            lines.append(f"  {role}: {m['content']}")

        lines.append("")
        lines.append("ANALYSIS:")

        if not analysis:
            lines.append("  [Analysis failed]")
            continue

        intent = analysis.get("intent", "unknown")
        satisfaction = analysis.get("satisfaction", "unknown")
        quality = analysis.get("quality_score", "?")
        mistakes = analysis.get("agent_mistakes", [])
        hidden = analysis.get("hidden_dissatisfaction", False)
        summary = analysis.get("summary", "")

        lines.append(f"  Intent: {intent}")
        lines.append(f"  Satisfaction: {satisfaction}")
        lines.append(f"  Quality Score: {quality}/5")
        lines.append(f"  Hidden Dissatisfaction: {'Yes' if hidden else 'No'}")

        if mistakes:
            lines.append(f"  Agent Mistakes: {', '.join(mistakes)}")
        else:
            lines.append("  Agent Mistakes: None")

        lines.append(f"  Summary: {summary}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    input_txt = sys.argv[1] if len(sys.argv) > 1 else "dialogues_batch.txt"
    output_json = sys.argv[2] if len(sys.argv) > 2 else "analysis_result.json"
    output_txt = sys.argv[3] if len(sys.argv) > 3 else "analysis_summary.txt"

    print(f"Reading dialogues from: {input_txt}")
    dialogues = parse_dialogues_txt(input_txt)
    print(f"Parsed {len(dialogues)} dialogues")

    analyses = []
    for i, dlg in enumerate(dialogues):
        print(f"[{i+1}/{len(dialogues)}] Analyzing seed={dlg['seed']}...")
        result = analyze_dialogue(dlg)
        analyses.append(result)
        if result:
            summary = result.get("summary", "")[:60]
            print(f"  ok: {summary}...")
        else:
            print("  FAILED after 3 attempts")

    # write JSON
    records = build_json_output(dialogues, analyses)
    output = {
        "model": MODEL,
        "count": len(records),
        "dialogues": records,
    }
    Path(output_json).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON -> {output_json}")

    # write summary TXT
    txt = build_summary_txt(dialogues, analyses)
    Path(output_txt).write_text(txt, encoding="utf-8")
    print(f"TXT  -> {output_txt}")


if __name__ == "__main__":
    main()
