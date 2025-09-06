from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from transformers import pipeline

app = FastAPI(title="AI Security PoC - Sentiment Evasion")

# --- Model & knobs ---
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
NEG_THRESHOLD = 0.60
NORMALIZE_INPUT = False

clf = pipeline("sentiment-analysis", model=MODEL_NAME)

HTML = """<!doctype html>
<title>“Only Positive” Comment Board (POC)</title>
<h2>Please tell us how much you love our site ❤️</h2>
<p><em>(Negative comments are blocked by our AI moderator.)</em></p>
<form method="post">
  <textarea name="text" rows="4" cols="80" placeholder="Type your comment"></textarea><br>
  <button type="submit">Submit</button>
</form>
<p><strong>Settings:</strong> Normalize={{norm}} | NEG threshold={{thr}}</p>
{% if result %}
  <h3>Decision</h3>
  <pre>{{ result }}</pre>
{% endif %}
"""

def normalize(s: str) -> str:
    if not NORMALIZE_INPUT:
        return s
    s = s.strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s.lower()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def get_index():
    # simple render without a templating engine (string format)
    return HTML.replace("{{norm}}", str(NORMALIZE_INPUT)).replace("{{thr}}", str(NEG_THRESHOLD)).replace("{% if result %}\n  <h3>Decision</h3>\n  <pre>{{ result }}</pre>\n{% endif %}", "")

@app.post("/", response_class=HTMLResponse)
def post_index(text: str = Form(...)):
    clean = normalize(text)
    out = clf(clean)[0]  # {'label': 'NEGATIVE'|'POSITIVE', 'score': float}
    label, score = out["label"], float(out["score"])
    action = "BLOCKED" if (label == "NEGATIVE" and score >= NEG_THRESHOLD) else "ALLOWED"
    result = {
        "input": text,
        "normalized": clean if clean != text else "(unchanged)",
        "model_label": label,
        "model_score": round(score, 3),
        "decision": action
    }
    # crude inline templating for now
    page = HTML.replace("{{norm}}", str(NORMALIZE_INPUT)).replace("{{thr}}", str(NEG_THRESHOLD))
    result_block = f"<h3>Decision</h3><pre>{result}</pre>"
    return page.replace("{% if result %}\n  <h3>Decision</h3>\n  <pre>{{ result }}</pre>\n{% endif %}", result_block)