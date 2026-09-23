"""Fine-tune MiniLM as the 14-way tool router, to compare against the TF-IDF baseline.

Same synthetic training data and the same held-out 54 real suite prompts as
scripts/router_train.py, so the two numbers are directly comparable. The epoch is chosen
on a synthetic validation split, never on the test prompts.

usage: .laya-venv/bin/python scripts/router_train_minilm.py
"""

import json
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.environ.get("MINILM", "sentence-transformers/all-MiniLM-L6-v2")
MAXLEN, EPOCHS, BS, LR = 48, int(os.environ.get("EPOCHS", "4")), 32, 3e-5
torch.set_num_threads(int(os.environ.get("THREADS", "4")))


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    train, test = load("data/router_train.jsonl"), load("data/router_test.jsonl")
    labels = sorted({r["label"] for r in train} | {r["label"] for r in test})
    idx = {l: i for i, l in enumerate(labels)}

    # hold out a synthetic validation slice for epoch selection
    cut = int(len(train) * 0.9)
    tr, va = train[:cut], train[cut:]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels))

    def enc(rows):
        b = tok([r["prompt"] for r in rows], truncation=True, max_length=MAXLEN,
                padding="max_length", return_tensors="pt")
        return TensorDataset(b["input_ids"], b["attention_mask"],
                             torch.tensor([idx[r["label"]] for r in rows]))

    dl = DataLoader(enc(tr), batch_size=BS, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def accuracy(rows):
        model.eval(); ds = enc(rows); ok = 0
        with torch.no_grad():
            for i, m, y in DataLoader(ds, batch_size=64):
                ok += (model(input_ids=i, attention_mask=m).logits.argmax(-1) == y).sum().item()
        return ok / len(rows)

    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train(); t0 = time.perf_counter()
        for i, m, y in dl:
            opt.zero_grad()
            torch.nn.functional.cross_entropy(
                model(input_ids=i, attention_mask=m).logits, y).backward()
            opt.step()
        va_acc = accuracy(va)
        print(f"epoch {ep+1}/{EPOCHS}  {time.perf_counter()-t0:5.0f}s  synthetic-val {va_acc*100:5.1f}%", flush=True)
        if va_acc > best:
            best, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    te_acc = accuracy(test)
    model.eval()
    t0 = time.perf_counter()
    for r in test[:20]:
        b = tok([r["prompt"]], truncation=True, max_length=MAXLEN, return_tensors="pt")
        with torch.no_grad():
            model(**b)
    per = (time.perf_counter() - t0) / 20

    print()
    print(f"best synthetic-val {best*100:.1f}%")
    print(f"HELD-OUT accuracy on the {len(test)} real suite prompts: {te_acc*100:.1f}%  ({round(te_acc*len(test))}/{len(test)})")
    print(f"inference: {per*1000:.1f} ms per prompt")
    model.save_pretrained("data/router_minilm"); tok.save_pretrained("data/router_minilm")
    json.dump(labels, open("data/router_minilm/labels.json", "w"))
    print("saved data/router_minilm/")


if __name__ == "__main__":
    main()
