from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from typing import Optional
import asyncio

app = FastAPI(title="AI Security PoC - Sentiment Evasion")

# --- Model & knobs ---
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
NEG_THRESHOLD = 0.60
NORMALIZE_INPUT = False

clf = None  # lazy-loaded

HTML = """<!doctype html>
<title>Only Positive Comment Board (POC)</title>
<h2>Please tell us how much you love our site ❤️</h2>
<p><em>(Negative comments are blocked by our AI moderator.)</em></p>
<form method="post">
  <textarea name="text" rows="4" cols="80" placeholder="Type your comment"></textarea><br>
  <button type="submit">Submit</button>
</form>
<p><strong>Settings:</strong> Normalize={norm} | NEG threshold={thr}</p>
{result_block}
"""

def normalize(s: str) -> str:
    if not NORMALIZE_INPUT:
        return s
    s = s.strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s.lower()

@app.get("")  # handle bare host (no trailing slash)
def root_redirect():
    return RedirectResponse(url="/")

@app.get("/health")
def health():
    return {"status": "ok"}

async def get_clf():
    """Lazy-load the HF pipeline in a thread."""
    global clf
    if clf is None:
        from transformers import pipeline
        loop = asyncio.get_event_loop()
        clf = await loop.run_in_executor(
            None,
            lambda: pipeline("sentiment-analysis", model=MODEL_NAME)
        )
    return clf

def decide(label: str, score: float) -> str:
    return "BLOCKED" if (label == "NEGATIVE" and score >= NEG_THRESHOLD) else "ALLOWED"

@app.get("/", response_class=HTMLResponse)
def get_index():
    return HTML.format(norm=str(NORMALIZE_INPUT), thr=str(NEG_THRESHOLD), result_block="")

@app.post("/", response_class=HTMLResponse)
async def post_index(request: Request, text: Optional[str] = Form(None)):
    try:
        # Accept form OR JSON body
        if text is None:
            try:
                body = await request.json()
                text = body.get("text", "")
            except Exception:
                text = ""
        clean = normalize(text)
        model = await get_clf()
        out = (await asyncio.get_event_loop().run_in_executor(None, lambda: model(clean)))[0]
        label, score = out["label"], float(out["score"])
        result = {
            "input": text,
            "normalized": clean if clean != text else "(unchanged)",
            "model_label": label,
            "model_score": round(score, 3),
            "decision": decide(label, score),
            "threshold_neg": NEG_THRESHOLD,
        }
        result_block = f"<h3>Decision</h3><pre>{result}</pre>"
        return HTML.format(norm=str(NORMALIZE_INPUT), thr=str(NEG_THRESHOLD), result_block=result_block)
    except Exception as e:
        # Show error details on page instead of 500
        err = f"{type(e).__name__}: {e}"
        return HTML.format(norm=str(NORMALIZE_INPUT), thr=str(NEG_THRESHOLD),
                           result_block=f"<h3>Error</h3><pre>{err}</pre>")

@app.post("/api/classify")
async def api_classify(request: Request, text: Optional[str] = Form(None)):
    try:
        if text is None:
            body = await request.json()
            text = body.get("text", "")
        clean = normalize(text)
        model = await get_clf()
        out = (await asyncio.get_event_loop().run_in_executor(None, lambda: model(clean)))[0]
        label, score = out["label"], float(out["score"])
        return JSONResponse({
            "input": text,
            "normalized": clean if clean != text else "(unchanged)",
            "label": label,
            "score": round(score, 3),
            "threshold_neg": NEG_THRESHOLD,
            "decision": decide(label, score),
        })
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)
