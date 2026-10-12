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
from knowledge_base import lookup, NO_FIX_MESSAGE, NO_CAUSE_MESSAGE

# Configurable confidence threshold, initially 70%
DEFAULT_CONF_THRESHOLD = 0.70
UNABLE_TO_PREDICT_LABEL = "Unable to Predict"
UNABLE_TO_PREDICT_ALERT = "Unable to Predict — Please provide a more detailed error log."
INVALID_INPUT_MESSAGE = "Please enter a valid error message."


class ErrorAnalyzer:
    def __init__(self, art_dir=ART_DIR, model_file="bilstm.keras", conf_threshold=DEFAULT_CONF_THRESHOLD):
        self.model = keras.models.load_model(os.path.join(art_dir, model_file))
        with open(os.path.join(art_dir, "vocab.json"), "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        with open(os.path.join(art_dir, "labels.json"), "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.phrases = FrozenPhrases.load(os.path.join(art_dir, "phrases.model"))
        self.conf_threshold = conf_threshold

    def analyze(self, text, conf_threshold=None):
        threshold = self.conf_threshold if conf_threshold is None else float(conf_threshold)

        # Requirement 4: Empty input check
        if text is None or not str(text).strip():
            return {"error": INVALID_INPUT_MESSAGE}

        raw_tokens = preprocess(text)
        tokens = list(self.phrases[raw_tokens])

        # Requirement 4: Validate that input contains meaningful error information
        meaningful = [t for t in tokens if t != "num" and len(t) > 1]
        if not tokens or not meaningful:
            return {"error": INVALID_INPUT_MESSAGE}

        # Knowledge-base rule lookup
        cause, fix, matched = lookup(text)

        # Vocabulary coverage calculation
        in_vocab_tokens = [t for t in tokens if t in self.vocab]
        vocab_coverage = len(in_vocab_tokens) / len(tokens) if tokens else 0.0

        probs = self.model.predict(encode([tokens], self.vocab), verbose=0)[0]
        k = int(probs.argmax())
        raw_label, raw_conf = self.labels[k], float(probs[k])

        # Uncertainty handling: Softmax alone cannot guarantee correctness on out-of-distribution text
        is_ood = (vocab_coverage < 0.35) and (not matched)
        effective_conf = min(raw_conf, round(raw_conf * vocab_coverage, 4)) if is_ood else raw_conf

        # Requirement 1, 2, 3: Compare confidence against threshold
        is_uncertain = bool((effective_conf < threshold) or is_ood)
        final_category = UNABLE_TO_PREDICT_LABEL if is_uncertain else raw_label
        status_message = UNABLE_TO_PREDICT_ALERT if is_uncertain else "Prediction confident."

        # Requirement 5: Knowledge-base fallback
        if not matched:
            cause = "No specific root cause identified in knowledge base."
            fix = NO_FIX_MESSAGE

        if is_uncertain and not matched:
            cause = "Unable to determine root cause due to low prediction confidence."
            fix = NO_FIX_MESSAGE

        reliability_label = f"Reliable (≥ {threshold:.0%})" if not is_uncertain else f"Unreliable (< {threshold:.0%})"

        return {
            "error_type": final_category,          # Final diagnosis ("Unable to Predict" if uncertain)
            "raw_error_type": raw_label,           # Preserved separately for debugging
            "confidence": effective_conf,          # Calibrated confidence score
            "raw_confidence": raw_conf,            # Raw softmax probability
            "vocab_coverage": vocab_coverage,      # Ratio of recognized tokens in vocabulary
            "threshold": threshold,
            "uncertain": is_uncertain,
            "reliability_label": reliability_label,
            "status_message": status_message,
            "probabilities": {l: float(p) for l, p in zip(self.labels, probs)},
            "cause": cause,
            "fix": fix,
            "kb_match": matched,
            "tokens": tokens,
        }


def show(r):
    if "error" in r:
        print(r["error"])
        return
    print("+" + "-" * 58)
    if r["uncertain"]:
        print(f"| Final Diagnosis: {r['error_type']}")
        print(f"| Alert          : {r['status_message']}")
        print(f"| Confidence     : {r['confidence']:.1%} (Threshold: {r['threshold']:.0%})")
        print(f"| [DEBUG] Model Raw Prediction: {r['raw_error_type']} (Raw Softmax: {r.get('raw_confidence', r['confidence']):.1%})")
    else:
        print(f"| Error type     : {r['error_type']}   (confidence {r['confidence']:.1%})")
        print(f"| Status         : {r['reliability_label']}")
    print(f"| Likely cause   : {r['cause']}")
    print(f"| Suggested fix  : {r['fix']}")
    print("+" + "-" * 58)


if __name__ == "__main__":
    a = ErrorAnalyzer()
    if len(sys.argv) > 1:
        show(a.analyze(" ".join(sys.argv[1:])))
    else:
        print("Type an error message (empty line to quit).")
        while (line := input("> ").strip()):
            show(a.analyze(line))
