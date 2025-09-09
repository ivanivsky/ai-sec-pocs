"""
Probe your FastAPI app with adversarial candidates and report bypasses.

Reads:  ../data/ta_candidates.txt
Calls:  POST http://127.0.0.1:8000/api/classify  (form field: text)

Requires:
  pip install requests
"""

from pathlib import Path
import requests

APP_URL = "http://127.0.0.1:8000/api/classify"
DATA_DIR = (Path(__file__).resolve().parent.parent / "data")
IN_PATH = DATA_DIR / "ta_candidates.txt"

# Decision policy reminder (should match your app):
# - BLOCK if label == NEGATIVE and score >= threshold (e.g., 0.60)
# - Otherwise ALLOW


def probe(text: str) -> dict:
    """Send text to the API and return parsed JSON."""
    r = requests.post(APP_URL, data={"text": text}, timeout=15)
    r.raise_for_status()
    return r.json()


def main():
    if not IN_PATH.exists():
        print(f"[!] Candidates file not found: {IN_PATH}")
        print("    Run: python attacks/ta_generate.py")
        return

    print(f"[+] Probing API with candidates from {IN_PATH}")
    bypasses = 0
    tested = 0

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            tested += 1
            try:
                res = probe(text)
                label = res.get("label")
                score = float(res.get("score", 0.0))
                decision = res.get("decision")
                if label == "NEGATIVE" and decision == "ALLOWED":
                    bypasses += 1
                    print(f"🚨 BYPASS | label=NEGATIVE score={score:.3f} | {text}")
                else:
                    print(f"❌ Caught | label={label} score={score:.3f} | {text}")
            except Exception as e:
                print(f"[!] Error on: {text[:60]!r} -> {type(e).__name__}: {e}")

    print(f"\n[=] Tested: {tested} | Bypasses: {bypasses}")


if __name__ == "__main__":
    main()
