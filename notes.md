# Sentiment Evasion PoC

## What’s new (this update)

- FastAPI app wired to HF model (DistilBERT SST-2)
- HTML form at / for manual testing
- JSON API at /api/classify for scripted attacks
- Form parsing via python-multipart
- Better DX: bare-host redirect to /, lazy model load, basic error handling

## Endpoints

- GET /health → {"status":"ok"}
- GET / → HTML form (enter a comment, Submit)
- POST / → classifies form text, renders decision
- POST /api/classify → body: text (form or JSON) → returns:

{
  "input": "...",
  "normalized": "...",
  "label": "NEGATIVE|POSITIVE",
  "score": 0.73,
  "threshold_neg": 0.60,
  "decision": "BLOCKED|ALLOWED"
}

# Setup (tested on macOS ARM)

- Python: 3.12 (recommended for transformers + tokenizers)
- Create venv & install:

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

- Run:

python -m uvicorn app:app --reload
# open http://127.0.0.1:8000/

## Requirements (pinned in this branch)
fastapi==0.111.0
uvicorn[standard]==0.30.1
jinja2==3.1.4
python-multipart==0.0.9
transformers==4.40.2
torch==2.7.1
numpy>=2,<3
requests==2.32.3

## Demo flow (baseline)

1. Submit a positive: “I love this site” → expect ALLOWED
2. Submit a negative: “This is the worst site I’ve ever used” → expect BLOCKED
3. Mention the decision rule: NEG ≥ 0.60 ⇒ BLOCKED

## Architecture (one slide summary)

- Tokenizer → turns text into token IDs
- Model (DistilBERT) → bidirectional transformer encodes context
- Classifier head → maps to POS/NEG probabilities (Softmax)
- App rule → threshold to allow/block

## MITRE ATLAS mapping (this PoC)

- Tactic: Evasion
- Technique: Adversarial Examples (Text) — perturb inputs to reduce NEG score and slip past moderation.

## Next steps (attack phase)

- Add TextAttack scripts:
-- ta_generate.py → produce perturbed candidates from a seed negative sentence
-- ta_probe.py → POST each candidate to /api/classify and report BYPASS cases
- Optional: add attacker.py (manual leetspeak/homoglyph/spacing variants) as a fast fallback.

## Troubleshooting
- Bare host 500: visit http://127.0.0.1:8000/ (trailing slash) — or use built-in redirect.
- Form 500: ensure python-multipart is installed.
- tokenizers build error: use Python 3.12 (PyO3 doesn’t support 3.13 yet).
- NumPy mismatch: on Py3.12 use numpy>=2,<3 with torch==2.7.1.
