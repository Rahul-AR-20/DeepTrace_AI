"""
Train + evaluate CNN, LSTM and Bi-LSTM on programming-error classification.

    python train.py                      # full run
    python train.py --epochs 5 --per-class 400   # quick smoke test

Outputs (in ./artifacts): bilstm.keras, lstm.keras, cnn.keras, vocab.json, labels.json,
phrases.model, word2vec.model, results.json and plots (confusion matrix, ROC, curves, comparison).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from common import ART_DIR, MAXLEN, OOV, PAD, encode, preprocess
from dataset import load_dataset

SEED = 42


# ----------------------------------------------------------------- models
def embedding(matrix, mask):
    return layers.Embedding(matrix.shape[0], matrix.shape[1],
                            embeddings_initializer=keras.initializers.Constant(matrix),
                            mask_zero=mask, trainable=True)  # Word2Vec init, fine-tuned


def build_bilstm(matrix, n_classes, stacked=False):
    inp = keras.Input(shape=(MAXLEN,))
    x = embedding(matrix, True)(inp)
    if stacked:   # optional "deep recurrent network"
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)       # forward + backward LSTM, concatenated
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="bilstm")


def build_lstm(matrix, n_classes):
    inp = keras.Input(shape=(MAXLEN,))
    x = embedding(matrix, True)(inp)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="lstm")


def build_cnn(matrix, n_classes):
    inp = keras.Input(shape=(MAXLEN,))
    x = embedding(matrix, False)(inp)
    x = layers.Conv1D(128, 3, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn")


def fit(model, Xtr, ytr, Xva, yva, epochs):
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    return model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=64,
                     callbacks=cb, verbose=2)


# ------------------------------------------------------------- evaluation
def evaluate(model, X, y, n_classes):
    probs = model.predict(X, verbose=0).astype("float64")
    probs /= probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(1)
    return dict(
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, average="macro", zero_division=0),
        recall=recall_score(y, pred, average="macro", zero_division=0),
        f1=f1_score(y, pred, average="macro", zero_division=0),
        roc_auc=roc_auc_score(y, probs, multi_class="ovr", average="macro",
                              labels=list(range(n_classes))),
    ), probs, pred


def plot_confusion(cm, labels, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (Bi-LSTM)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y, probs, labels, path):
    Y = label_binarize(y, classes=range(len(labels)))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, name in enumerate(labels):
        fpr, tpr, _ = roc_curve(Y[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (Bi-LSTM, one-vs-rest)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(hist, path):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    axs[0].plot(hist.history["loss"], label="train"); axs[0].plot(hist.history["val_loss"], label="val")
    axs[0].set_title("Loss"); axs[0].legend()
    axs[1].plot(hist.history["accuracy"], label="train"); axs[1].plot(hist.history["val_accuracy"], label="val")
    axs[1].set_title("Accuracy"); axs[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_comparison(results, path):
    names = list(results)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    w = 0.25
    for k, metric in enumerate(["accuracy", "f1", "roc_auc"]):
        ax.bar(np.arange(len(names)) + k * w, [results[n][metric] for n in names], w, label=metric)
    ax.set_xticks(np.arange(len(names)) + w, names)
    ax.set_ylim(0.5, 1.02); ax.set_title("Model comparison (test set)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--per-class", type=int, default=1200)
    ap.add_argument("--emb", type=int, default=100)
    ap.add_argument("--stacked", action="store_true", help="use 2 stacked Bi-LSTM layers")
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    df = load_dataset(args.per_class, SEED)
    labels = sorted(df.label.unique())
    y = df.label.map({l: i for i, l in enumerate(labels)}).values
    print(f"{len(df)} messages, classes: {labels}")

    # 2. NLP preprocessing --------------------------------------------------
    tokens = [preprocess(m) for m in df.message]
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)

    # 3. n-grams (bigram detection learned on the TRAIN split only) ----------
    phrases = Phrases([tokens[i] for i in tr], min_count=5, threshold=8.0, delimiter="_").freeze()
    tokens = [phrases[t] for t in tokens]

    # 4. Word2Vec (trained on the TRAIN split only) --------------------------
    w2v = Word2Vec([tokens[i] for i in tr], vector_size=args.emb, window=5, min_count=1,
                   sg=1, epochs=50, workers=1, seed=SEED)
    vocab = {w: i + 2 for i, w in enumerate(w2v.wv.index_to_key)}   # 0=PAD, 1=OOV
    matrix = np.random.RandomState(SEED).normal(0, 0.05, (len(vocab) + 2, args.emb)).astype("float32")
    matrix[PAD] = 0
    for w, i in vocab.items():
        matrix[i] = w2v.wv[w]
    print(f"vocab size: {len(vocab)}; example similar to 'null':",
          [w for w, _ in w2v.wv.most_similar("null", topn=4)] if "null" in w2v.wv else "n/a")

    X = encode(tokens, vocab)
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    # 5. train the three models ----------------------------------------------
    n = len(labels)
    models = {"CNN": build_cnn(matrix, n), "LSTM": build_lstm(matrix, n),
              "Bi-LSTM": build_bilstm(matrix, n, args.stacked)}
    results, hist_bi = {}, None
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        hist = fit(model, Xtr, ytr, Xva, yva, args.epochs)
        res, probs, pred = evaluate(model, Xte, yte, n)
        results[name] = res
        if name == "Bi-LSTM":
            hist_bi, probs_bi, pred_bi = hist, probs, pred
        model.save(os.path.join(ART_DIR, name.lower().replace("-", "") + ".keras"))

    # 6. report ----------------------------------------------------------------
    print("\n=== Test-set comparison ===")
    print(f"{'model':<9}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'ROC-AUC':>9}")
    for k, r in results.items():
        print(f"{k:<9}{r['accuracy']:8.4f}{r['precision']:8.4f}{r['recall']:8.4f}{r['f1']:8.4f}{r['roc_auc']:9.4f}")
    print("\nBi-LSTM classification report:")
    print(classification_report(yte, pred_bi, target_names=labels, digits=4))
    cm = confusion_matrix(yte, pred_bi)
    print("Confusion matrix:\n", cm)

    plot_confusion(cm, labels, os.path.join(ART_DIR, "confusion_matrix.png"))
    plot_roc(yte, probs_bi, labels, os.path.join(ART_DIR, "roc_curves.png"))
    plot_history(hist_bi, os.path.join(ART_DIR, "training_curves.png"))
    plot_comparison(results, os.path.join(ART_DIR, "model_comparison.png"))

    # 7. save artifacts for inference ------------------------------------------
    json.dump(vocab, open(os.path.join(ART_DIR, "vocab.json"), "w"))
    json.dump(labels, open(os.path.join(ART_DIR, "labels.json"), "w"))
    json.dump(results, open(os.path.join(ART_DIR, "results.json"), "w"), indent=2)
    phrases.save(os.path.join(ART_DIR, "phrases.model"))
    w2v.save(os.path.join(ART_DIR, "word2vec.model"))
    print(f"\nSaved everything to {ART_DIR}")


if __name__ == "__main__":
    main()
