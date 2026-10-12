"""Shared NLP preprocessing (used by BOTH training and inference so they always match)."""
import os
import re

import numpy as np
from nltk.stem import PorterStemmer

MAXLEN = 40          # max tokens per error message
PAD, OOV = 0, 1      # reserved ids
ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

# Small stop-word list. Words like "no", "not", "cannot" are deliberately KEPT
# because they carry meaning in error messages.
STOP = {"the", "a", "an", "of", "to", "in", "at", "on", "is", "are", "was", "be", "by",
        "with", "and", "or", "it", "this", "that", "for", "as"}

_stem = PorterStemmer().stem
_camel = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_tok = re.compile(r"[a-z]+|\d+|[;(){}\[\]]")   # words, numbers, and syntax-relevant punctuation


def preprocess(text, stem=True):
    """
    'java.lang.NullPointerException: Cannot invoke ...'
        -> ['java', 'lang', 'null', 'pointer', 'except', 'cannot', 'invok', ...]
    Steps: camelCase split -> lowercase -> tokenize -> stop-word removal
           -> number normalisation -> stemming
    """
    text = _camel.sub(" ", str(text)).lower()
    out = []
    for t in _tok.findall(text):
        if t.isdigit():
            t = "num"
        elif t in STOP:
            continue
        elif stem and len(t) > 3:
            t = _stem(t)
        out.append(t)
    return out


def encode(token_lists, word2idx, maxlen=MAXLEN):
    """Token lists -> padded integer matrix (post-padding with 0)."""
    X = np.zeros((len(token_lists), maxlen), dtype="int32")
    for i, toks in enumerate(token_lists):
        ids = [word2idx.get(t, OOV) for t in toks][:maxlen]
        X[i, :len(ids)] = ids
    return X
