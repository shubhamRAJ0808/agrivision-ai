"""
Trains LeafDiseaseCNN from scratch on the dataset in ./data (4 tomato leaf classes
sampled from the public PlantVillage dataset). Saves checkpoints/disease_model.pt,
checkpoints/labels.json and checkpoints/metrics.json.

Run: python3 train.py
"""
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LeafDataset
from model import LeafDiseaseCNN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")
EPOCHS = 12
BATCH_SIZE = 32
LR = 1e-3


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = LeafDataset.from_folder(DATA_DIR, val_split=0.15)
    print(f"Classes: {train_ds.classes}")
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LeafDiseaseCNN(num_classes=len(train_ds.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = 0.0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
        })
        print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val_loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "disease_model.pt"))

    elapsed = time.time() - start
    with open(os.path.join(CKPT_DIR, "labels.json"), "w") as f:
        json.dump(train_ds.classes, f, indent=2)
    with open(os.path.join(CKPT_DIR, "metrics.json"), "w") as f:
        json.dump({
            "history": history,
            "best_val_acc": round(best_val_acc, 4),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "classes": train_ds.classes,
            "training_seconds": round(elapsed, 1),
            "note": "Trained from scratch (no pretrained weights available in this sandbox) "
                    "on a small sample of the public PlantVillage dataset. Accuracy is honest "
                    "and reflects the small data/compute budget, not a production-scale model.",
        }, f, indent=2)
    print(f"\nDone in {elapsed:.1f}s. Best val acc: {best_val_acc:.3f}")
    print(f"Saved model to {CKPT_DIR}/disease_model.pt")


if __name__ == "__main__":
    main()
