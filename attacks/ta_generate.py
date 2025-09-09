# ta_generate.py
"""
Generate adversarial variations of a seed negative sentence
using TextAttack (TextFooler recipe).
"""

from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def main():
    # Load model + tokenizer into a TextAttack wrapper
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Build attack recipe
    attack = TextFoolerJin2019.build(wrapper)

    # Seed example: negative sentiment
    input_text = "This is the worst site I have ever used."
    dataset = Dataset([[input_text, 0]])  # 0 = negative label in SST-2

    # Run attack
    for result in attack.attack_dataset(dataset):
        print(result)  # show summary
        with open("ta_candidates.txt", "w") as f:
            for pr in result.perturbed_results:
                f.write(pr.attacked_text.text + "\n")

if __name__ == "__main__":
    main()
