"""
DeepTrace AI - Neural Error Classifier & Diagnostic Inference:
    python predict.py "ModuleNotFoundError: No module named 'pandas'"
    python predict.py            (interactive mode)
"""
import json
import os
import sys

from gensim.models.phrases import FrozenPhrases
from tensorflow import keras

from common import ART_DIR, encode, preprocess
from knowledge_base import lookup

CONF_THRESHOLD = 0.60   # below this the answer is flagged as uncertain


class ErrorAnalyzer:
    def __init__(self, art_dir=ART_DIR, model_file="bilstm.keras"):
        self.model = keras.models.load_model(os.path.join(art_dir, model_file))
        self.vocab = json.load(open(os.path.join(art_dir, "vocab.json")))
        self.labels = json.load(open(os.path.join(art_dir, "labels.json")))
        self.phrases = FrozenPhrases.load(os.path.join(art_dir, "phrases.model"))

    def analyze(self, text):
        tokens = self.phrases[preprocess(text)]
        if not tokens:
            return {"error": "Please enter an error message."}
        probs = self.model.predict(encode([tokens], self.vocab), verbose=0)[0]
        k = int(probs.argmax())
        label, conf = self.labels[k], float(probs[k])
        cause, fix, matched = lookup(text, label)
        return {
            "error_type": label,
            "confidence": conf,
            "uncertain": conf < CONF_THRESHOLD,
            "probabilities": {l: float(p) for l, p in zip(self.labels, probs)},
            "cause": cause,
            "fix": fix,
            "kb_match": matched,
            "tokens": tokens,
        }


def show(r):
    if "error" in r:
        print(r["error"]); return
    print("+" + "-" * 58)
    print(f"| Error type : {r['error_type']}   (confidence {r['confidence']:.1%})"
          + ("   [LOW CONFIDENCE]" if r["uncertain"] else ""))
    print(f"| Likely cause: {r['cause']}")
    print(f"| Suggested fix: {r['fix']}")
    print("+" + "-" * 58)


if __name__ == "__main__":
    a = ErrorAnalyzer()
    if len(sys.argv) > 1:
        show(a.analyze(" ".join(sys.argv[1:])))
    else:
        print("Type an error message (empty line to quit).")
        while (line := input("> ").strip()):
            show(a.analyze(line))
