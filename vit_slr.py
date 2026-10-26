"""
Alcohol detection from iris / periocular NIR images
ViT + SimCLR (self-supervised pretraining) -> supervised fine-tuning -> unseen-image prediction

Expected folder layout (labels = sub-folder names):

    split2/
      train/  alcoholic/   non_alcoholic/
      val/    alcoholic/   non_alcoholic/
      test/   alcoholic/   non_alcoholic/

USAGE
-----
pip install torch torchvision timm scikit-learn matplotlib pillow numpy

1) python alcohol_vit_simclr.py pretrain                  # SimCLR on train images (labels ignored)
2) python alcohol_vit_simclr.py finetune                  # supervised fine-tuning + temperature calibration
3) python alcohol_vit_simclr.py evaluate                  # accuracy etc. on split2/test (or --folder ANY_LABELLED_FOLDER)
4) python alcohol_vit_simclr.py predict --image my_eye.png          # single unseen image
   python alcohol_vit_simclr.py predict --image some_folder/        # many unseen images (+ accuracy if the
                                                                     # folder has alcoholic/non_alcoholic sub-folders)
   python alcohol_vit_simclr.py all                       # steps 1-3 in one go

Use --data_root "/path/to/split2" if your dataset is somewhere else.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import timm
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DEFAULT_ROOT = Path("/Users/spartan/Desktop/major_project_alcohol_detection/dataset_collection/split2")
CLASSES = ["non_alcoholic", "alcoholic"]          # index 0 = non_alcoholic, 1 = alcoholic
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARCH = "vit_small_patch16_224"
IMG = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("."))


def load_img(path):
    return Image.open(path).convert("RGB")      # NIR gray -> 3 channels for the ViT


# --------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------
simclr_aug = T.Compose([
    T.RandomResizedCrop(IMG, scale=(0.4, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(0.4, 0.4, 0.2),
    T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
train_aug = T.Compose([
    T.RandomResizedCrop(IMG, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(8),
    T.ColorJitter(0.2, 0.2),
    T.ToTensor(), T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
class SimCLRSet(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = load_img(self.paths[i])
        return simclr_aug(im), simclr_aug(im)


class LabelledSet(Dataset):
    """Reads <folder>/<class_name>/*.png ; returns (tensor, label, path)."""

    def __init__(self, folder, tf):
        self.items = []
        for idx, c in enumerate(CLASSES):
            d = Path(folder) / c
            if d.is_dir():
                self.items += [(p, idx) for p in list_images(d)]
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(load_img(p)), y, str(p)


class PathSet(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return eval_tf(load_img(self.paths[i])), -1, str(self.paths[i])


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class SimCLRModel(nn.Module):
    def __init__(self, imagenet=False):
        super().__init__()
        self.encoder = timm.create_model(ARCH, pretrained=imagenet, num_classes=0)
        d = self.encoder.num_features
        self.proj = nn.Sequential(nn.Linear(d, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))

    def forward(self, x):
        return self.proj(self.encoder(x))


class Classifier(nn.Module):
    def __init__(self, imagenet=False):
        super().__init__()
        self.encoder = timm.create_model(ARCH, pretrained=imagenet, num_classes=0)
        self.head = nn.Linear(self.encoder.num_features, 2)

    def forward(self, x):
        return self.head(self.encoder(x))


def nt_xent(z1, z2, temp=0.5):
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / temp
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


# --------------------------------------------------------------------------------------
# Step 1: SimCLR pre-training (labels are NOT used)
# --------------------------------------------------------------------------------------
def pretrain(a):
    dev = get_device()
    paths = list_images(Path(a.data_root) / "train")
    if a.unlabeled_dir:
        paths += list_images(a.unlabeled_dir)       # optional extra unlabeled images
    assert paths, f"No images found under {a.data_root}/train"
    print(f"[SimCLR] {len(paths)} images, device={dev}")

    loader = DataLoader(SimCLRSet(paths), batch_size=a.batch, shuffle=True, drop_last=True,
                        num_workers=a.workers)
    model = SimCLRModel(a.imagenet).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr_pre, weight_decay=0.05)
    total = a.pre_epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))

    Path(a.out).mkdir(exist_ok=True, parents=True)
    for ep in range(a.pre_epochs):
        model.train(); run = 0.0
        for v1, v2 in loader:
            loss = nt_xent(model(v1.to(dev)), model(v2.to(dev)), a.temp)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += loss.item()
        print(f"epoch {ep + 1}/{a.pre_epochs}  loss {run / len(loader):.4f}")
        torch.save(model.encoder.state_dict(), Path(a.out) / "simclr_encoder.pth")
    print("Saved:", Path(a.out) / "simclr_encoder.pth")


# --------------------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------------------
@torch.no_grad()
def get_logits(model, loader, dev):
    model.eval(); L, Y, P = [], [], []
    for x, y, p in loader:
        L.append(model(x.to(dev)).cpu()); Y += list(y.numpy() if torch.is_tensor(y) else y); P += list(p)
    return torch.cat(L), np.array(Y), P


def fit_temperature(logits, labels):
    """Temperature scaling on the validation set -> calibrated confidence."""
    labels = torch.tensor(labels)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp().item())


# --------------------------------------------------------------------------------------
# Step 2: supervised fine-tuning
# --------------------------------------------------------------------------------------
def finetune(a):
    dev = get_device()
    tr = LabelledSet(Path(a.data_root) / "train", train_aug)
    va = LabelledSet(Path(a.data_root) / "val", eval_tf)
    assert len(tr) and len(va), "train/ or val/ folder is empty - check --data_root"
    counts = np.bincount([y for _, y in tr.items], minlength=2)
    print(f"[Fine-tune] train {len(tr)} (non_alc={counts[0]}, alc={counts[1]}), val {len(va)}")

    model = Classifier(a.imagenet)
    enc_path = Path(a.out) / "simclr_encoder.pth"
    if enc_path.exists() and not a.imagenet:
        model.encoder.load_state_dict(torch.load(enc_path, map_location="cpu"))
        print("Loaded SimCLR encoder")
    else:
        print("WARNING: no SimCLR encoder loaded (run `pretrain` first) - training from current init")
    model.to(dev)

    w = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)           # handles class imbalance
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers)
    vl = DataLoader(va, batch_size=a.batch, num_workers=a.workers)

    def make_opt(freeze):
        for p in model.encoder.parameters():
            p.requires_grad = not freeze
        groups = [{"params": model.head.parameters(), "lr": 1e-3}]
        if not freeze:
            groups.append({"params": model.encoder.parameters(), "lr": a.lr_ft})
        return torch.optim.AdamW(groups, weight_decay=0.05)

    opt = make_opt(freeze=True)
    sched = None
    best, bad = -1, 0
    for ep in range(a.ft_epochs):
        if ep == a.probe_epochs:                    # switch from linear probe to full fine-tune
            opt = make_opt(freeze=False)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.ft_epochs - a.probe_epochs)
        model.train(); run = 0.0
        for x, y, _ in tl:
            loss = crit(model(x.to(dev)), y.to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); run += loss.item()
        if sched:
            sched.step()
        logits, ys, _ = get_logits(model, vl, dev)
        acc = accuracy_score(ys, logits.argmax(1).numpy())
        print(f"epoch {ep + 1}/{a.ft_epochs}  train loss {run / len(tl):.4f}  val acc {acc:.4f}")
        if acc > best:
            best, bad = acc, 0
            torch.save(model.state_dict(), Path(a.out) / "classifier_best.pth")
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping"); break

    model.load_state_dict(torch.load(Path(a.out) / "classifier_best.pth", map_location=dev))
    logits, ys, _ = get_logits(model, vl, dev)
    T_ = fit_temperature(logits, ys)
    json.dump({"temperature": T_, "best_val_acc": best}, open(Path(a.out) / "calibration.json", "w"))
    print(f"Best val acc {best:.4f} | temperature {T_:.3f} | saved to {a.out}")


# --------------------------------------------------------------------------------------
# Load trained model
# --------------------------------------------------------------------------------------
def load_trained(a):
    dev = get_device()
    model = Classifier(False)
    model.load_state_dict(torch.load(Path(a.out) / "classifier_best.pth", map_location="cpu"))
    model.to(dev).eval()
    cal = Path(a.out) / "calibration.json"
    T_ = json.load(open(cal))["temperature"] if cal.exists() else 1.0
    return model, T_, dev


# --------------------------------------------------------------------------------------
# Step 3: EVALUATION  (accuracy check on labelled unseen images)
# --------------------------------------------------------------------------------------
def evaluate(a):
    model, T_, dev = load_trained(a)
    folder = Path(a.folder) if a.folder else Path(a.data_root) / "test"
    ds = LabelledSet(folder, eval_tf)
    assert len(ds), f"No images in {folder}/(alcoholic|non_alcoholic)"
    logits, ys, paths = get_logits(model, DataLoader(ds, batch_size=a.batch, num_workers=a.workers), dev)
    probs = F.softmax(logits / T_, dim=1).numpy()
    pred = probs.argmax(1)

    acc = accuracy_score(ys, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(ys, pred, pos_label=1, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(ys, probs[:, 1])
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(ys, pred, labels=[0, 1])

    print("\n================ EVALUATION ON", folder, "================")
    print(f"Images      : {len(ys)}")
    print(f"ACCURACY    : {acc * 100:.2f}%")
    print(f"Precision   : {pr * 100:.2f}%   Recall: {rc * 100:.2f}%   F1: {f1 * 100:.2f}%   ROC-AUC: {auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred) [non_alcoholic, alcoholic]:\n", cm)
    print(classification_report(ys, pred, target_names=CLASSES, zero_division=0))

    # accuracy when low-confidence images are rejected as "Uncertain"
    conf = probs.max(1)
    keep = conf >= a.threshold
    if keep.any():
        print(f"With reject option (confidence >= {a.threshold}): coverage {keep.mean() * 100:.1f}%, "
              f"accuracy on accepted {accuracy_score(ys[keep], pred[keep]) * 100:.2f}%")

    Path(a.out).mkdir(exist_ok=True, parents=True)
    with open(Path(a.out) / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "true", "pred", "confidence", "p_alcoholic"])
        for p, y, pr_, pb in zip(paths, ys, pred, probs):
            w.writerow([p, CLASSES[y], CLASSES[pr_], f"{pb.max():.4f}", f"{pb[1]:.4f}"])
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], CLASSES, rotation=20); ax.set_yticks([0, 1], CLASSES)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.tight_layout()
        plt.savefig(Path(a.out) / "confusion_matrix.png", dpi=150)
    except Exception as e:
        print("Plot skipped:", e)
    print("Saved predictions + confusion matrix in", a.out)


# --------------------------------------------------------------------------------------
# Step 4: PREDICT unseen image(s)
# --------------------------------------------------------------------------------------
def predict(a):
    model, T_, dev = load_trained(a)
    target = Path(a.image)
    paths = [target] if target.is_file() else list_images(target)
    assert paths, f"No images found at {target}"
    logits, _, plist = get_logits(model, DataLoader(PathSet(paths), batch_size=a.batch), dev)
    probs = F.softmax(logits / T_, dim=1).numpy()

    rows = []
    print("\n================ PREDICTION ================")
    for p, pb in zip(plist, probs):
        k = int(pb.argmax()); conf = float(pb[k])
        label = CLASSES[k].upper() if conf >= a.threshold else "UNCERTAIN (low confidence - recheck / other iris variation)"
        rows.append((p, label, conf, float(pb[1])))
        print(f"{Path(p).name:40s} -> {label}   confidence {conf * 100:.1f}%  (P(alcoholic) = {pb[1] * 100:.1f}%)")
    if target.is_dir():
        with open(Path(a.out) / "unseen_predictions.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["path", "prediction", "confidence", "p_alcoholic"]); w.writerows(rows)
        print("Saved", Path(a.out) / "unseen_predictions.csv")
        # If the folder is labelled (alcoholic/ non_alcoholic sub-folders) also report real accuracy
        if any((target / c).is_dir() for c in CLASSES):
            a.folder = str(target); evaluate(a)
    print("\nNOTE: 'confidence' is the model's calibrated probability, not guaranteed accuracy.\n"
          "Real accuracy is measured by `evaluate` on labelled unseen images.")


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pretrain", "finetune", "evaluate", "predict", "all"])
    ap.add_argument("--data_root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--image", help="image file or folder for `predict`")
    ap.add_argument("--folder", help="labelled folder (with alcoholic/ non_alcoholic/) for `evaluate`")
    ap.add_argument("--unlabeled_dir", help="optional extra unlabeled images for SimCLR")
    ap.add_argument("--imagenet", action="store_true", help="start ViT from ImageNet weights (needs internet)")
    ap.add_argument("--pre_epochs", type=int, default=100)
    ap.add_argument("--ft_epochs", type=int, default=30)
    ap.add_argument("--probe_epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr_pre", type=float, default=3e-4)
    ap.add_argument("--lr_ft", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.70, help="below this confidence -> UNCERTAIN")
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    seed_all()

    if a.mode == "pretrain":
        pretrain(a)
    elif a.mode == "finetune":
        finetune(a)
    elif a.mode == "evaluate":
        evaluate(a)
    elif a.mode == "predict":
        assert a.image, "give --image path"
        predict(a)
    else:
        pretrain(a); finetune(a); evaluate(a)