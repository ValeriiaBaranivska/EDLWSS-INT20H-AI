"""
export_txt.py — Export generated dialogues to human-readable TXT format.

Usage:
    python export_txt.py output/final_10/ output/final_10/dialogues.txt
"""

import json
from pathlib import Path


def export_txt(input_dir: str, output_path: str):
    in_dir = Path(input_dir)
    files = sorted(in_dir.glob("dialogue_*.json"))
    lines = []

    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))

        p = d["params"]
        lines.append("=" * 60)
        lines.append(f"DIALOGUE  seed={p['seed']}  #{f.stem.split('_')[-1]}")
        lines.append(f"complexity : {p['complexity']}")
        lines.append(f"sector     : {p['sector']}")
        lines.append(f"topic      : {p['topic']}")
        lines.append(f"outcome    : {p['outcome']}")
        lines.append(f"style      : {p['style']}")
        lines.append(f"client     : {p['client_archetype']}")
        lines.append(f"agent      : {p['agent_archetype']}")
        lines.append(f"twist      : {p['twist']}")
        lines.append(f"use_dolphin: {p['use_dolphin']}")
        lines.append("")

        # characters
        ch = d.get("characters", {})
        cl = ch.get("client", {})
        ag = ch.get("agent", {})
        if cl.get("name") not in (None, "", "Customer"):
            lines.append(
                f"Client : {cl['name']} | "
                f"mood={cl.get('mood', '')} | "
                f"{cl.get('backstory', '')[:60]}"
            )
            lines.append(f"Agent  : {ag['name']} | mood={ag.get('mood', '')}")
            lines.append("")

        # messages
        for m in d["messages"]:
            role = m["role"]
            turn = m.get("turn", "?")
            content = m["content"]
            emotion = m.get("emotion") or ""
            label = "CLIENT" if role == "client" else "AGENT "
            em_tag = f"  [{emotion}]" if emotion else ""
            lines.append(f"[{label} turn={turn}]{em_tag}")
            lines.append(f"  {content}")
            lines.append("")

        # meta
        meta = d.get("meta", {})
        lines.append(f"stages : {meta.get('stages_run', [])}")
        lines.append(f"dirt   : {meta.get('dirt_applied', [])}")
        lines.append(f"time   : {meta.get('total_time_s', '?')}s")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(files)} dialogues -> {output_path}")


if __name__ == "__main__":
    import sys

    in_dir = sys.argv[1] if len(sys.argv) > 1 else "output/final_10"
    out_txt = sys.argv[2] if len(sys.argv) > 2 else "output/final_10/dialogues.txt"
    export_txt(in_dir, out_txt)
