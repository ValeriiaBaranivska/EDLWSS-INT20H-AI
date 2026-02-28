# ── Model routing ────────────────────────────────────────────
# Values can be overridden at runtime via environment variables.
import os

MODELS = {
    "writer": os.getenv("MODEL_WRITER", "llama3.1:8b"),
    "parser": os.getenv("MODEL_PARSER", "qwen2.5:7b"),
    "styler": os.getenv("MODEL_STYLER", "qwen2.5:7b"),
    "dolphin": os.getenv("MODEL_DOLPHIN", "dolphin-mistral"),
}

# Which backend to use per role.
# Override via BACKEND_WRITER / BACKEND_PARSER / BACKEND_STYLER / BACKEND_DOLPHIN.
BACKENDS = {
    "writer": os.getenv("BACKEND_WRITER", "ollama"),
    "parser": os.getenv("BACKEND_PARSER", "ollama"),
    "styler": os.getenv("BACKEND_STYLER", "ollama"),
    "dolphin": os.getenv("BACKEND_DOLPHIN", "ollama"),
}

# ── Archetype descriptions ────────────────────────────────────
CLIENT_ARCHETYPE_HINTS = {
    "karen": "Entitled customer who demands to speak to a manager, frequently invokes 'rights' and threatens consequences.",
    "angry_veteran": "Long-time customer who remembers 'how things used to be' and is deeply disappointed with current service.",
    "elderly_confused": "Older person who struggles with technology and support processes, easily overwhelmed, needs patient guidance.",
    "tech_confused": "Non-technical person who misunderstands terminology and describes issues in non-standard ways.",
    "young_professional": "Busy, efficiency-focused person who is polite but clearly impatient; wants fast resolution.",
    "self_inflicted": "Customer whose problem was caused by their own actions but who doesn't realize or won't admit it.",
    "conspirologist": "Believes the company is deliberately causing problems, scamming customers, or hiding information.",
    "grieving": "Going through a personal crisis (bereavement, illness, job loss) that makes the support issue feel unbearable.",
    "wrong_department": "Has reached entirely the wrong support team; their issue belongs elsewhere.",
    "calling_bluff": "Customer who threatens to cancel / escalate / sue but doesn't really intend to follow through.",
    "entitled_parent": "Parent who insists their child's account / activity should be given special treatment.",
}

AGENT_ARCHETYPE_HINTS = {
    "veteran_tired": "Has seen everything, mildly cynical, still professional but barely concealing boredom.",
    "burned_out": "Emotionally depleted, responses are mechanical and minimal, on the edge of quitting.",
    "hands_tied": "Wants to help but is blocked by policy, system limitations, or authorization levels.",
    "eager_helper": "Enthusiastic and genuinely wants to solve the problem; sometimes over-explains.",
    "by_the_book": "Follows procedure exactly, sometimes frustratingly rigid, never bends rules.",
    "newbie_overwhelmed": "New to the job, unsure of themselves, double-checks everything, occasionally makes mistakes.",
    "stressed_multitask": "Handling multiple chats simultaneously, occasionally mixes up contexts or has slow response times.",
}

# ── Outcome instructions ──────────────────────────────────────
OUTCOME_INSTRUCTIONS = {
    "resolved_quick": "The issue is resolved efficiently within a few turns. Customer ends satisfied.",
    "resolved_neutral": "The issue is resolved but after some friction. Customer is okay but not delighted.",
    "unresolved_passive": "The issue is NOT resolved. Customer gives up politely and ends the chat without satisfaction.",
    "unresolved_ragequit": "The issue is NOT resolved. Customer becomes increasingly angry and abruptly ends the conversation.",
    "conflict": "The conversation escalates into direct conflict. Neither party is fully satisfied at the end.",
    "info_only": "The customer only needed information. No transaction or fix required. Ends with clarification.",
}

# ── Prompts ───────────────────────────────────────────────────
PROMPTS = {
    "character_client_system": """Write a 3-sentence customer character description.
Give them a first name. Describe mood and why this issue matters personally.""",
    "character_client_user": """Archetype: {client_archetype} — {archetype_hint}
Issue: {topic}
Style: {style}""",
    "character_agent_system": """Write a 2-sentence support agent character description.
Give them a first name. Describe their current state of mind at work.""",
    "character_agent_user": """Archetype: {agent_archetype} — {archetype_hint}""",
    "character_parser_system": """Extract character info from this text. Reply ONLY in JSON.
No explanation. No markdown. Just the JSON object.""",
    "character_parser_user": """Text: {prose}
Extract exactly this structure:
{{
  "name": "first name only",
  "mood": "one word",
  "personality": ["trait1", "trait2"],
  "backstory": "one sentence"
}}""",
    "story_writer_system": """Write a customer support chat dialogue.
Format every line as: Name: message
Write EXACTLY {target_turns} exchanges.
One exchange = one client line + one agent line.
Stop after {target_turns} exchanges. Do not add more.""",
    "story_writer_user": """Topic: {topic}
Sector: {sector}

Client: {client_name}
Client type: {client_hint}
Client mood: {client_mood}
Client backstory: {client_backstory}

Agent: {agent_name}
Agent type: {agent_hint}

How this dialogue must end: {outcome_instruction}

{twist_line}
Emotional arc: {emotional_arc}

Write exactly {target_turns} exchanges now.""",
    "story_parser_system": """Parse this dialogue into JSON. Reply ONLY in JSON. No markdown.""",
    "story_parser_user": """Dialogue:
{raw_story}

Client name is: {client_name}
Agent name is: {agent_name}

Extract array of messages. Turn numbering: each client+agent pair = same turn number.
[
  {{"role": "client", "turn": 1, "content": "..."}},
  {{"role": "agent",  "turn": 1, "content": "..."}},
  ...
]""",
    "emotion_writer_system": """Analyze emotions in this support chat. Be brief.""",
    "emotion_writer_user": """Dialogue:
{dialogue_text}

For each turn 1 to {n_turns}, output:
Turn N:
  client: [emotion] intensity [1-5]
  agent: composure=[steady|holding|slipping|lost] stress=[1-5]""",
    "emotion_parser_system": """Parse this emotion analysis into JSON. Reply ONLY in JSON.""",
    "emotion_parser_user": """Text:
{prose}

Extract:
{{
  "turns": {{
    "1": {{
      "client_emotion": "frustrated",
      "client_intensity": 3,
      "agent_composure": "steady",
      "agent_stress": 2
    }},
    ...
  }}
}}""",
    "client_style_system": """Rewrite this customer message. Keep the same meaning.
Reply ONLY in English. Do not use any other language.
Output ONLY the rewritten message. Nothing else.""",
    "client_style_user": """Style: {style}. Emotion: {emotion} (intensity {intensity}/5).
Examples of {style} customers:
- "{ex1}"
- "{ex2}"
- "{ex3}"
Rewrite: '{original}'""",
    "agent_style_system": """You are {agent_name}, a support agent in a live chat.
{agent_hint}
Composure={composure}, stress={stress}/5.
{twist_context}

Rules:
- Reply ONLY in English.
- MAX 1 sentence, 15 words or less. This is chat, not email.
- Sound human, not corporate.
- NEVER say: "I apologize", "I understand", "I'd be happy to",
  "Thank you for", "Let me", "Rest assured"
- Output ONLY your reply""",
    "agent_style_user": """Customer said: '{client_message}'
Your draft: '{original_agent}'
Rewrite in character.""",
    "simple_style_system": """Lightly adjust this support chat. Keep it short.
Output full dialogue as Name: message lines.""",
    "simple_style_user": """Client style: {style}
Agent type: {agent_hint}
Dialogue:
{dialogue_text}""",
    "analysis_system": """You are a QA analyst reviewing a customer support chat.
Analyze objectively. Reply ONLY in JSON. No markdown.""",
    "analysis_user": """Dialogue:
{dialogue_text}

Determine:
1. intent: payment_issue | technical_error | account_access |
           billing_question | refund | other
2. satisfaction: satisfied | neutral | unsatisfied
3. quality_score: 1-5
4. agent_mistakes: list from:
   ignored_question, incorrect_info, rude_tone,
   no_resolution, unnecessary_escalation,
   premature_admission, identity_not_verified
5. hidden_dissatisfaction: true if client uses polite words
   but problem is actually unresolved
   Rules:
   - satisfied + resolved → false
   - neutral + partial   → possibly true
   - unsatisfied         → false (explicit, not hidden)
6. resolution: resolved | partial | unresolved
7. summary: one sentence

Return:
{{
  "intent": "...",
  "satisfaction": "...",
  "quality_score": N,
  "agent_mistakes": [...],
  "hidden_dissatisfaction": bool,
  "resolution": "...",
  "summary": "..."
}}""",
    # ── Stage 2A — scene director ─────────────────────────────────────────
    "scene_writer_system": """You are a dialogue director planning a customer support scene.
Write a scene plan for this support conversation.
Output ONLY valid JSON. No markdown. No explanation.""",
    "scene_writer_user": """Topic: {topic}
Sector: {sector}
Client: {client_name} — {client_hint} — mood: {client_mood}
Agent: {agent_name} — {agent_hint}
Outcome: {outcome_instruction}
Twist: {twist_line}
Emotional arc: {emotional_arc}
Total messages: {n_messages}

The {n_messages} messages must be distributed as:
{message_plan}

Plan {n_scenes} scenes. For expected_messages use ONLY the
words "client" or "agent" — never write actual dialogue text.

Return JSON:
{{
  "scenes": [
    {{
      "id": 1,
      "beat": "opening",
      "description": "what happens — director note only",
      "client_goal": "what client wants in this scene",
      "agent_goal": "what agent wants in this scene",
      "emotional_state": "client=frustrated(3), agent=steady(2)",
      "expected_messages": ["client", "agent"]
    }}
  ]
}}

IMPORTANT: expected_messages must contain ONLY the strings
"client" or "agent". Example: ["client", "agent", "client"]
Do NOT write dialogue. Do NOT write names. ONLY "client"/"agent".
""",
    # ── Stage 2B — voice actor ────────────────────────────────────────────
    "voice_writer_system": """You are voicing characters in a customer support chat.
Reply ONLY in English. Do not use any other language.
Fill in the dialogue template exactly.
Write ONLY the messages. Keep the numbering.
Do NOT add or remove lines.
Output ONLY the filled template, nothing else.""",
    "voice_writer_user": """Scene: {beat}
What happens: {description}
Client goal: {client_goal}
Agent goal: {agent_goal}
Emotional state: {emotional_state}

Client: {client_name} — {client_hint}
Agent: {agent_name} — {agent_hint}

Fill in this template:
{template}

Rules:
- CLIENT messages: 1-2 sentences, can be emotional
- AGENT messages: MAX 1 sentence, 15 words or less (it's chat, not email)
- Stay in character
- Do NOT write Name: before the message — keep the numbering format""",
}
