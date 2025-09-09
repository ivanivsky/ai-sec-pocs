"""
Generate adversarial variants for a seed negative sentence using TextAttack.
Writes candidates to ../data/ta_candidates.txt

Requires:
  pip install textattack==0.3.8 transformers==4.40.2
"""

from pathlib import Path
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Target model: same as your app
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Seed input (negative sentiment). You can change this for demos.
SEED_TEXT = "This is the worst site I have ever used."
# For SST-2, label mapping is: 0 = NEGATIVE, 1 = POSITIVE
SEED_LABEL = 0

# Where to save candidates (../data relative to this file)
DATA_DIR = (Path(__file__).resolve().parent.parent / "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = DATA_DIR / "ta_candidates.txt"


def main():
    print(f"[+] Loading model/tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    print("[+] Building TextFooler attack")
    attack = TextFoolerJin2019.build(wrapper)

    print(f"[+] Attacking seed: {SEED_TEXT!r}")
    dataset = Dataset([[SEED_TEXT, SEED_LABEL]])

    n_total = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for result in attack.attack_dataset(dataset):
            # result has .original_result and .perturbed_results
            for pr in result.perturbed_results:
                text = pr.attacked_text.text
                # Dedup nearby variants
                if not text.strip():
                    continue
                f.write(text.replace("\n", " ").strip() + "\n")
                n_total += 1

    print(f"[+] Wrote {n_total} candidates to {OUT_PATH}")


if __name__ == "__main__":
    main()
