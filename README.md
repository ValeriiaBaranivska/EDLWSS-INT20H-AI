# Customer Support Dialogue Generator & Analyzer

Automated system for generating and analyzing customer support dialogues using LLMs.

## Overview

This project generates realistic customer support chat dialogues and analyzes them for quality assessment. It identifies:
- **Intent** of customer inquiries
- **Customer satisfaction** level
- **Agent quality score**
- **Agent mistakes** (if any)
- **Hidden dissatisfaction** (formal thanks but problem unresolved)

## Features

### Dialogue Generation (`generate.py`)
- Generates diverse customer support scenarios:
  - Payment issues
  - Technical errors
  - Account access problems
  - Billing questions
  - Refund requests
- Includes various outcomes:
  - Successful resolutions
  - Problematic cases
  - Conflict scenarios
  - Agent mistakes

### Dialogue Analysis (`analyze.py`)
- Uses GLM-4.7 model for analysis
- Returns structured JSON with:
  - `intent`: payment_issue | technical_error | account_access | billing_question | refund_request | other
  - `satisfaction`: satisfied | neutral | unsatisfied
  - `quality_score`: 1-5 scale
  - `agent_mistakes`: List of detected issues
  - `hidden_dissatisfaction`: Boolean flag

## Installation

```bash
# Clone the repository
git clone https://github.com/ValeriiaBaranivska/EDLWSS-INT20H-AI.git
cd EDLWSS-INT20H-AI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Dialogues

```bash
# Generate 10 dialogues (default)
python generate.py

# Generate specific number of dialogues
python generate.py 20

# With custom seed
python generate.py 10 12345
```

Output: `dialogues_batch.txt` and `dialogues_batch.json`

### Analyze Dialogues

```bash
# Analyze with default files
python analyze.py

# Specify input/output files
python analyze.py dialogues_batch.txt analysis_result.json analysis_summary.txt
```

Output:
- `analysis_result.json` — structured JSON with all analysis data
- `analysis_summary.txt` — human-readable report

## Models Used

- **Generation**: Llama 3.1 8B (local via Ollama) + Qwen 2.5 7B
- **Analysis**: GLM-4.7 (via Z.AI API)

## Requirements

- Python 3.11+
- Ollama (for local LLM inference)
- API key for GLM-4.7 (Z.AI)

## Project Structure

```
├── generate.py          # Dialogue generation script
├── analyze.py           # Dialogue analysis script
├── requirements.txt     # Python dependencies
├── dialogues_batch.txt  # Generated dialogues (sample)
├── dialogues_batch.json # Generated dialogues with metadata
├── analysis_result.json # Analysis results
├── analysis_summary.txt # Human-readable analysis report
└── support-dialogue-gen/
    ├── core/            # Core modules
    ├── pipeline/        # Generation pipeline stages
    └── corpus/          # Data banks (fillers, slang)
```

## Sample Output

### Generated Dialogue
```
Client: I need this fixed immediately or I'll cancel my account.
Agent: Um, can you tell me more about the issue?
Client: The bill is incorrect; it's OVERCHARGED by $50!
Agent: Uh, i see. Let's get that adjusted right away.
```

### Analysis Result
```json
{
  "intent": "payment_issue",
  "satisfaction": "neutral",
  "quality_score": 4,
  "agent_mistakes": [],
  "hidden_dissatisfaction": false,
  "summary": "Client threatens cancellation over $50 overcharge. Agent responds professionally and offers quick resolution."
}
```

## Authors

- INT20H AI Team

## License

MIT
