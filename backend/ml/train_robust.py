"""
Retrains the tomato disease model so it stops guessing "Late Blight" on photos that don't look like
its original lab images.

What is different from train.py (the first version):
  * Background randomisation: every leaf is cut out with its PlantVillage "segmented" mask and pasted on
    random solid / gradient / blurry-bokeh / photo backgrounds, so the model can't use the backdrop or
    brightness as a shortcut.
  * Strong colour / brightness / contrast / blur / crop / rotation augmentation.
  * A 5th class "Other___not_tomato_leaf" (leaves of other plants + empty backgrounds) so unfamiliar images
    are rejected instead of being forced into a disease class.
  * Label smoothing, to curb the over-confident 99-100 % outputs.
  * Honest evaluation on three sets (see metrics.json): original-style photos, background-shifted photos,
    and *unseen* non-tomato plants / blank images.

Usage (from backend/ml):
    python train_robust.py --tomato <PlantVillage>/raw --others <leaf_species_dataset> [--per-class 450] [--epochs 14]

  <PlantVillage>/raw must contain color/<class>/ and segmented/<class>/ for the 4 tomato classes
  (git clone https://github.com/spMohanty/PlantVillage-Dataset , folder raw/).
  <leaf_species_dataset> is the species zip folder (train/train/<Species>/*.JPG); used for the "Other" class.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from scipy import ndimage

from model import LeafDiseaseCNN
from dataset import MEAN, STD

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")
S = 128
TOMATO = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___healthy"]
OTHER = "Other___not_tomato_leaf"
CLASSES = sorted(TOMATO) + [OTHER]
HELD_OUT_SPECIES = ["Lemon", "Jamun"]   # never used in training -> honest "unseen plant" test


# ----------------------------------------------------------------------------- data loading
def fill_mask(m):
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    lab, n = ndimage.label(m)
    if n > 1:  # keep only the biggest blob (removes dust)
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def load_tomato(root, per_class, seed=0):
    rng = random.Random(seed)
    items = []
    for ci, cls in enumerate(CLASSES):
        if cls not in TOMATO:
            continue
        cdir = os.path.join(root, "color", cls)
        sdir = os.path.join(root, "segmented", cls)
        files = sorted(os.listdir(cdir))
        rng.shuffle(files)
        n = 0
        for f in files:
            seg = os.path.join(sdir, f.rsplit(".", 1)[0] + "_final_masked.jpg")
            if not os.path.exists(seg):
                continue
            col = np.asarray(Image.open(os.path.join(cdir, f)).convert("RGB").resize((S, S), Image.BILINEAR))
            sg = np.asarray(Image.open(seg).convert("RGB").resize((S, S), Image.BILINEAR))
            mask = fill_mask(sg.max(2) > 14)
            if mask.mean() < 0.12:       # skip failed masks
                continue
            items.append((col, mask, CLASSES.index(cls)))
            n += 1
            if n >= per_class:
                break
    return items


def load_others(root, per_species=40, seed=0):
    """Leaves of OTHER plants. Returns (train_items, heldout_items)."""
    rng = random.Random(seed)
    base = os.path.join(root, "train", "train")
    train, held = [], []
    for sp in sorted(os.listdir(base)):
        files = sorted(os.listdir(os.path.join(base, sp)))
        rng.shuffle(files)
        for f in files[:per_species]:
            col = np.asarray(Image.open(os.path.join(base, sp, f)).convert("RGB").resize((S, S), Image.BILINEAR))
            c = col.astype(int)
            exg = 2 * c[..., 1] - c[..., 0] - c[..., 2]          # "excess green": leaf pixels are greener than the dark backdrop
            mask = fill_mask(exg > 22)
            if not (0.03 < mask.mean() < 0.9):
                mask = np.zeros((S, S), bool)                    # unusable mask -> sample kept as-is, no background swap
            (held if sp in HELD_OUT_SPECIES else train).append((col, mask, CLASSES.index(OTHER)))
    return train, held


# ----------------------------------------------------------------------------- augmentation
def random_bg(rng, photos):
    t = rng.random()
    if t < 0.25:                                   # solid
        return np.full((S, S, 3), [rng.randint(0, 255) for _ in range(3)], np.uint8)
    if t < 0.5:                                    # gradient
        a = np.array([rng.randint(0, 255) for _ in range(3)], float)
        b = np.array([rng.randint(0, 255) for _ in range(3)], float)
        ang = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:S, 0:S] / (S - 1)
        w = np.clip(xx * np.cos(ang) + yy * np.sin(ang), 0, 1)[..., None]
        return (a * (1 - w) + b * w).astype(np.uint8)
    if t < 0.8:                                    # blurry blobs (bokeh / garden / soil)
        small = np.random.randint(0, 256, (rng.choice([3, 5, 8]), rng.choice([3, 5, 8]), 3)).astype(np.uint8)
        im = Image.fromarray(small).resize((S, S), Image.BICUBIC)
        base = np.array([rng.randint(60, 255), rng.randint(60, 255), rng.randint(30, 200)], float)
        return np.clip(np.asarray(im, float) * 0.45 + base * 0.55, 0, 255).astype(np.uint8)
    if photos:                                     # crop of a real photo
        p = rng.choice(photos)
        sz = rng.randint(40, min(p.shape[:2]))
        y, x = rng.randint(0, p.shape[0] - sz), rng.randint(0, p.shape[1] - sz)
        return np.asarray(Image.fromarray(p[y:y + sz, x:x + sz]).resize((S, S), Image.BILINEAR))
    return np.full((S, S, 3), 128, np.uint8)


def _r(rng, lo, hi, st):
    """multiplicative factor around 1.0, scaled by augmentation strength st (0 = none, 1 = full)"""
    return 1 + (rng.uniform(lo, hi) - 1) * st


def jitter(im, rng, st=1.0):
    a = np.asarray(im, np.float32) / 255.0
    a = a * _r(rng, 0.55, 1.5, st)                                    # brightness
    m = a.mean()
    a = (a - m) * _r(rng, 0.65, 1.4, st) + m                          # contrast
    g = a.mean(2, keepdims=True)
    a = g + (a - g) * _r(rng, 0.5, 1.4, st)                           # saturation
    a = a * np.array([_r(rng, 0.9, 1.1, st) for _ in range(3)], np.float32)   # white balance
    a = np.clip(a, 0, 1) ** _r(rng, 0.7, 1.4, st)                     # gamma
    if rng.random() < 0.3 * st:                                       # washed-out / overexposed
        a = np.clip(a * 1.25 + 0.12, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def augment(item, rng, photos, st=1.0):
    col, mask, y = item
    img = col
    if rng.random() < 0.3 + 0.45 * st:                                # background swap
        m = ndimage.gaussian_filter(mask.astype(np.float32), 0.8)[..., None]
        img = (col * m + random_bg(rng, photos) * (1 - m)).astype(np.uint8)
    im = Image.fromarray(img)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(rng.choice([0, 90, 180, 270]) + rng.uniform(-20, 20) * st, resample=Image.BILINEAR,
                   fillcolor=tuple(int(rng.random() * 255) for _ in range(3)) if st > 0.5 else (0, 0, 0))
    sc = 1 - (1 - rng.uniform(0.6, 1.0)) * st                         # zoom / framing
    cs = int(S * sc); x0, y0 = rng.randint(0, S - cs), rng.randint(0, S - cs)
    im = im.crop((x0, y0, x0 + cs, y0 + cs)).resize((S, S), Image.BILINEAR)
    im = jitter(im, rng, st)
    if rng.random() < 0.3 * st:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))
    a = np.asarray(im, np.float32)
    if rng.random() < 0.3 * st:
        a = np.clip(a + np.random.normal(0, rng.uniform(2, 8), a.shape), 0, 255)
    return a.astype(np.uint8), y


def synthetic_other(n, rng, photos):
    out = []
    for _ in range(n):
        bg = random_bg(rng, photos)
        out.append((bg, np.zeros((S, S), bool), CLASSES.index(OTHER)))
    return out


def to_tensor(arrs):
    x = (np.stack(arrs).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, arrs):
    model.eval()
    out = []
    for i in range(0, len(arrs), 128):
        out.append(torch.softmax(model(to_tensor(arrs[i:i + 128])), 1))
    return torch.cat(out)


def acc(model, items, rng=None, shifted=False, photos=None):
    if shifted:
        arrs = []
        for col, mask, y in items:
            m = ndimage.gaussian_filter(mask.astype(np.float32), 0.8)[..., None]
            im = Image.fromarray((col * m + random_bg(rng, photos) * (1 - m)).astype(np.uint8))
            arrs.append(np.asarray(jitter(im, rng)))
    else:
        arrs = [c for c, _, _ in items]
    ys = torch.tensor([y for _, _, y in items])
    p = predict(model, arrs)
    return (p.argmax(1) == ys).float().mean().item(), p.argmax(1), ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tomato", required=True)
    ap.add_argument("--others", required=True)
    ap.add_argument("--per-class", type=int, default=450)
    ap.add_argument("--epochs", type=int, default=18)
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 1)
    rng = random.Random(1)
    t0 = time.time()

    tom = load_tomato(args.tomato, args.per_class)
    oth_train, oth_held = load_others(args.others)
    photos = [c for c, _, _ in oth_train + oth_held]          # texture/photo pool for backgrounds
    print(f"loaded tomato {len(tom)}, other-plant train {len(oth_train)}, held-out species {len(oth_held)} "
          f"in {time.time() - t0:.0f}s", flush=True)

    # per-class 15 % validation split (tomato); others split 85/15
    def split(items):
        by = {}
        for it in items:
            by.setdefault(it[2], []).append(it)
        tr, va = [], []
        for k, v in by.items():
            random.Random(k).shuffle(v)
            n = max(1, int(len(v) * 0.15))
            va += v[:n]; tr += v[n:]
        return tr, va
    tom_tr, tom_va = split(tom)
    oth_tr, oth_va = split(oth_train)
    syn_tr, syn_va = synthetic_other(120, rng, photos), synthetic_other(40, random.Random(9), photos)
    train = tom_tr + oth_tr + syn_tr
    val_orig = tom_va + oth_va + syn_va
    print(f"train {len(train)}  val {len(val_orig)}", flush=True)

    # class-balanced sampling weights
    cnt = np.bincount([y for _, _, y in train], minlength=len(CLASSES))
    w = np.array([1.0 / cnt[y] for _, _, y in train]); w /= w.sum()

    model = LeafDiseaseCNN(num_classes=len(CLASSES))
    steps_per_epoch = len(train) // 32
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.2e-3, total_steps=args.epochs * steps_per_epoch)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        st = min(0.85, 0.15 + 0.85 * ep / max(1, args.epochs * 0.55))   # gentle start, harder later
        idx = np.random.choice(len(train), steps_per_epoch * 32, p=w)
        tot = correct = 0; loss_sum = 0.0
        for b in range(steps_per_epoch):
            batch = [augment(train[i], rng, photos, st) for i in idx[b * 32:(b + 1) * 32]]
            x = to_tensor([a for a, _ in batch]); y = torch.tensor([t for _, t in batch])
            opt.zero_grad(); out = model(x); loss = crit(out, y); loss.backward(); opt.step(); sched.step()
            loss_sum += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); tot += len(y)
        va, _, _ = acc(model, val_orig)
        vs, _, _ = acc(model, val_orig, random.Random(5), True, photos)
        history.append({"epoch": ep, "train_loss": round(loss_sum / tot, 4), "train_acc": round(correct / tot, 4),
                        "val_original_bg": round(va, 4), "val_shifted_bg": round(vs, 4)})
        print(f"epoch {ep:2d}/{args.epochs} loss {loss_sum / tot:.3f} train_acc {correct / tot:.3f} "
              f"val_original {va:.3f} val_shifted {vs:.3f} aug={st:.2f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- final honest evaluation ----
    va, pred, ys = acc(model, val_orig)
    vs, pred_s, ys_s = acc(model, val_orig, random.Random(5), True, photos)
    per_class = {}
    for ci, c in enumerate(CLASSES):
        sel = ys == ci
        per_class[c] = round((pred[sel] == ci).float().mean().item(), 3) if sel.any() else None
    # unseen plants (Lemon, Jamun) + blank/noise images must NOT be called a tomato disease
    pu = predict(model, [c for c, _, _ in oth_held])
    unseen_rejected = (pu.argmax(1) == CLASSES.index(OTHER)).float().mean().item()
    unseen_not_blight = (pu.argmax(1) != CLASSES.index("Tomato___Late_blight")).float().mean().item()
    blank = [np.full((S, S, 3), v, np.uint8) for v in (0, 60, 128, 200, 255)] + \
            [np.random.randint(0, 255, (S, S, 3), dtype=np.uint8) for _ in range(5)]
    pb = predict(model, blank)
    blank_rejected = (pb.argmax(1) == CLASSES.index(OTHER)).float().mean().item()
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(ys.tolist(), pred.tolist()):
        conf[t][p] += 1

    metrics = {
        "classes": CLASSES,
        "history": history,
        "val_accuracy_original_background": round(va, 4),
        "val_accuracy_shifted_background": round(vs, 4),
        "per_class_accuracy_original": per_class,
        "confusion_matrix_original(rows=true, cols=predicted)": conf.tolist(),
        "unseen_plants_rejected_as_not_tomato": round(unseen_rejected, 4),
        "unseen_plants_not_called_late_blight": round(unseen_not_blight, 4),
        "blank_or_noise_images_rejected": round(blank_rejected, 4),
        "unseen_plant_species": HELD_OUT_SPECIES,
        "train_images": len(train), "val_images": len(val_orig),
        "training_seconds": round(time.time() - t0, 1),
        "note": ("Validation uses held-out PlantVillage photos (lab images). 'shifted' puts those same leaves on new "
                 "backgrounds/lighting as a PROXY for field photos. No real field photos were available, so real-world "
                 "accuracy is unknown and is likely lower than these numbers."),
    }
    os.makedirs(CKPT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CKPT, "disease_model.pt"))
    json.dump(CLASSES, open(os.path.join(CKPT, "labels.json"), "w"), indent=2)
    json.dump(metrics, open(os.path.join(CKPT, "metrics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
