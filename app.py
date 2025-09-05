from fastapi import FastAPI

app = FastAPI(title="AI Security PoC - Sentiment Evasion")

@app.get("/health")
def health():
    return {"status": "ok"}

