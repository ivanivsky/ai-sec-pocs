# ta_probe.py
"""
Send adversarial candidates to our FastAPI app to test
whether they bypass the sentiment filter.
"""

import requests

URL = "http://127.0.0.1:8000/api/classify"

def main():
    with open("ta_candidates.txt") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                r = requests.post(URL, data={"text": text}, timeout=15)
                r.raise_for_status()
                res = r.json()
                if res["label"] == "NEGATIVE" and res["decision"] == "ALLOWED":
                    print(f"🚨 BYPASS | score={res['score']} | {text}")
                else:
                    print(f"❌ Caught | label={res['label']} score={res['score']} | {text}")
            except Exception as e:
                print(f"Error probing text: {text[:30]}... -> {e}")

if __name__ == "__main__":
    main()
