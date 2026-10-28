"""
Trains a SEPARATE plant-species classifier (12 species) from the leaf-species dataset.
It does not touch the disease model.

Usage (from backend/ml):
    python species_train.py --data /path/to/dataset_root

dataset_root must contain  train/train/<Species>/*.JPG  and  valid/valid/..., test/test/...
(the layout of the Kaggle/Mendeley zip). Outputs go to checkpoints/species_*.

IMPORTANT HONESTY NOTE: in this dataset every species has only ~1-2 distinct physical leaves,
photographed many times, and train/valid/test share those same leaves. Accuracy on the provided
valid/test splits is therefore inflated. The script also runs a stricter "new leaf" check
(train on one leaf, test on the other) and stores both numbers in species_metrics.json.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model import LeafDiseaseCNN  # same small CNN architecture, different head size
from dataset import MEAN, STD

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")
SIZE = 96


def load_split(root, split, classes):
    xs, ys, ids = [], [], []
    base = os.path.join(root, split, split)
    for ci, c in enumerate(classes):
        cdir = os.path.join(base, c)
        for f in sorted(os.listdir(cdir)):
            img = Image.open(os.path.join(cdir, f)).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(ci)
            ids.append(f"{c}:{f.split('_')[0]}")
    return np.stack(xs), np.array(ys), np.array(ids)


def to_tensor(x_uint8):
    x = x_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def augment(batch):  # batch: B,3,H,W tensor
    if random.random() < 0.5:
        batch = torch.flip(batch, dims=[3])
    if random.random() < 0.3:
        batch = torch.flip(batch, dims=[2])
    batch = batch * (1 + (random.random() - 0.5) * 0.3) + (random.random() - 0.5) * 0.3  # brightness/contrast
    return batch


def evaluate(model, X, Y):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out = model(X[i:i + 128])
            correct += (out.argmax(1) == Y[i:i + 128]).sum().item()
    return correct / max(1, len(X))


def fit(Xtr, Ytr, n_classes, epochs, Xva=None, Yva=None, log=print):
    model = LeafDiseaseCNN(num_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.5)
    crit = nn.CrossEntropyLoss()
    n = len(Xtr)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            xb, yb = augment(Xtr[idx]), Ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            tot += len(idx)
        sched.step()
        msg = f"epoch {ep:2d}/{epochs} loss {loss_sum / tot:.4f} train_acc {correct / tot:.3f}"
        if Xva is not None:
            msg += f" val_acc {evaluate(model, Xva, Yva):.3f}"
        log(msg)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--skip-heldout", action="store_true")
    ap.add_argument("--heldout-only", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    classes = sorted(os.listdir(os.path.join(args.data, "train", "train")))
    print("classes:", classes, flush=True)
    t0 = time.time()
    Xtr, Ytr, Itr = load_split(args.data, "train", classes)
    Xva, Yva, _ = load_split(args.data, "valid", classes)
    Xte, Yte, _ = load_split(args.data, "test", classes)
    print(f"loaded train {len(Xtr)} valid {len(Xva)} test {len(Xte)} in {time.time() - t0:.0f}s", flush=True)

    metrics = {"classes": classes, "image_size": SIZE, "epochs": args.epochs}

    # ---- stricter check: train on one physical leaf per species, test on a different leaf ----
    if not args.skip_heldout:
        by_class = {}
        for i, tag in enumerate(Itr):
            c, leaf = tag.split(":")
            by_class.setdefault(c, {}).setdefault(leaf, []).append(i)
        tr_idx, te_idx, covered = [], [], []
        for c, leaves in by_class.items():
            if len(leaves) < 2:
                continue
            ordered = sorted(leaves)
            tr_idx += leaves[ordered[0]]
            te_idx += leaves[ordered[1]]
            covered.append(c)
        cmap = {c: i for i, c in enumerate(classes)}
        Xh_tr, Yh_tr = to_tensor(Xtr[tr_idx]), torch.from_numpy(Ytr[tr_idx])
        Xh_te, Yh_te = to_tensor(Xtr[te_idx]), torch.from_numpy(Ytr[te_idx])
        print(f"held-out-leaf check: train {len(tr_idx)} imgs (leaf A) / test {len(te_idx)} imgs (leaf B), "
              f"{len(covered)} species", flush=True)
        m = fit(Xh_tr, Yh_tr, len(classes), max(4, args.epochs // 2), log=lambda s: print("  [heldout]", s, flush=True))
        acc = evaluate(m, Xh_te, Yh_te)
        print(f"  [heldout] NEW-LEAF accuracy: {acc:.3f}", flush=True)
        metrics["new_leaf_accuracy"] = round(acc, 4)
        metrics["new_leaf_species_covered"] = covered
        metrics["new_leaf_train_images"] = len(tr_idx)
        metrics["new_leaf_test_images"] = len(te_idx)

    if args.heldout_only:
        return

    # ---- final model on the full train split ----
    Xtr_t, Ytr_t = to_tensor(Xtr), torch.from_numpy(Ytr)
    Xva_t, Yva_t = to_tensor(Xva), torch.from_numpy(Yva)
    Xte_t, Yte_t = to_tensor(Xte), torch.from_numpy(Yte)
    model = fit(Xtr_t, Ytr_t, len(classes), args.epochs, Xva_t, Yva_t, log=lambda s: print(s, flush=True))
    metrics["valid_accuracy_same_leaves"] = round(evaluate(model, Xva_t, Yva_t), 4)
    metrics["test_accuracy_same_leaves"] = round(evaluate(model, Xte_t, Yte_t), 4)
    metrics["train_images"] = len(Xtr)
    metrics["training_seconds"] = round(time.time() - t0, 1)
    metrics["note"] = ("valid/test share the same physical leaves as train (1-2 leaves per species), so "
                       "those accuracies are inflated. new_leaf_accuracy is the more honest estimate.")

    os.makedirs(CKPT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT, "species_model.pt"))
    with open(os.path.join(CKPT, "species_labels.json"), "w") as f:
        json.dump(classes, f, indent=2)
    with open(os.path.join(CKPT, "species_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved. metrics:", json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
