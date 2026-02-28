import json
from pathlib import Path

# Read analysis
with open('analysis_result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = []
lines.append('=' * 70)
lines.append('CUSTOMER SUPPORT DIALOGUE ANALYSIS SUMMARY')
lines.append(f'Model: {data["model"]}')
lines.append(f'Total dialogues analyzed: {data["count"]}')
lines.append('=' * 70)
lines.append('')

for d in data['dialogues']:
    lines.append('-' * 70)
    lines.append(f'DIALOGUE seed={d["seed"]} | {d["topic"]} | {d["outcome"]} | {d["style"]}')
    lines.append('-' * 70)
    lines.append('')
    lines.append('CONVERSATION:')
    for m in d['messages']:
        role = 'Client' if m['role'] == 'client' else 'Agent'
        lines.append(f'  {role}: {m["content"]}')
    lines.append('')
    lines.append('ANALYSIS:')
    dlg = d['dialogue']
    lines.append(f'  Summary: {dlg["summary"]}')
    lines.append(f'  Intent: {d["topic"]}')

    # Determine satisfaction
    sat = dlg['client_satisfaction_estimate']
    if sat >= 0.7:
        satisfaction = 'satisfied'
    elif sat >= 0.4:
        satisfaction = 'neutral'
    else:
        satisfaction = 'unsatisfied'
    lines.append(f'  Satisfaction: {satisfaction} ({sat:.1f})')

    # Quality score 1-5
    eff = dlg['agent_effectiveness']
    emp = dlg['agent_empathy']
    quality = round((eff + emp) / 2 * 5)
    quality = max(1, min(5, quality))
    lines.append(f'  Quality Score: {quality}/5')

    # Agent mistakes
    mistakes = []
    if emp < 0.4:
        mistakes.append('lack_of_empathy')
    if eff < 0.3:
        mistakes.append('no_resolution')
    if dlg['resolution_quality'] == 'poor':
        mistakes.append('slow_response')
    if dlg['conflict_level'] == 'high' and dlg['emotional_arc'] != 'deescalates':
        mistakes.append('unnecessary_escalation')

    lines.append(f'  Agent Mistakes: {mistakes if mistakes else "none"}')

    # Hidden dissatisfaction
    hidden = dlg['resolution_quality'] in ['poor', 'mediocre'] and sat >= 0.3
    lines.append(f'  Hidden Dissatisfaction: {hidden}')
    lines.append(f'  Emotional Arc: {dlg["emotional_arc"]}')
    lines.append(f'  Conflict Level: {dlg["conflict_level"]}')
    lines.append('')

Path('summary.txt').write_text('\n'.join(lines), encoding='utf-8')
print('Created summary.txt')
