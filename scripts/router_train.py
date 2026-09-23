"""Train a tool-routing classifier and score it on the held-out real suite prompts.

Baseline first: TF-IDF (word + char n-grams) into logistic regression. If this clears the
bar it replaces Laya's forward pass with a few milliseconds of numpy and removes torch
(~2.2 GB RSS) from the deployment entirely.

usage: uv run --with scikit-learn --python 3.14 python scripts/router_train.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    train, test = load("data/router_train.jsonl"), load("data/router_test.jsonl")
    Xtr = [r["prompt"] for r in train]; ytr = [r["label"] for r in train]
    Xte = [r["prompt"] for r in test];  yte = [r["label"] for r in test]

    model = Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t0

    pred = model.predict(Xte)
    acc = float(np.mean([p == y for p, y in zip(pred, yte)]))

    # latency, one prompt at a time, as the pipeline would call it
    t0 = time.perf_counter()
    for x in Xte: model.predict([x])
    per = (time.perf_counter() - t0) / len(Xte)

    print(f"trained on {len(Xtr)} synthetic rows in {fit_s:.1f}s")
    print(f"HELD-OUT accuracy on the {len(Xte)} real suite prompts: {acc*100:.1f}%  ({sum(p==y for p,y in zip(pred,yte))}/{len(Xte)})")
    print(f"inference: {per*1000:.2f} ms per prompt")
    print()
    print("errors:")
    for r, p in zip(test, pred):
        if p != r["label"]:
            print(f"  {r['id']:38s} want {r['label']:24s} got {p}")
    Path("data").mkdir(exist_ok=True)
    import pickle
    pickle.dump(model, open("data/router_tfidf.pkl", "wb"))
    print("\nsaved data/router_tfidf.pkl")


if __name__ == "__main__":
    main()
