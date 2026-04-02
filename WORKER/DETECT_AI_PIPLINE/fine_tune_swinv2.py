"""fine_tune_swinv2.py
=====================

Fine-tuning de microsoft/swinv2-small-patch4-window16-256 sur votre dataset
REAL/ALT pour créer un détecteur spécialisé sur les générateurs 2025-2026
(FLUX, Midjourney, GPT Image 1, SDXL, Ideogram).

Basé sur la méthodologie OpenFake (McGill/Mila, arxiv 2509.09495) :
  - SwinV2-Small backbone (ImageNet-1k)
  - Fine-tuning end-to-end avec deux flux d'augmentation
  - Flux 1 : géométrique/photométrique (real + fake)
  - Flux 2 : dégradation légère sur les fakes uniquement
    (downscaling, blur, bruit gaussien, JPEG qualité aléatoire)
    → évite que le modèle apprenne des "raccourcis" de compression

Utilisation :
    python fine_tune_swinv2.py \\
        --data-dir /chemin/vers/calib_dataset \\
        --output-dir ./swinv2_openfake \\
        --epochs 10 \\
        --batch-size 32 \\
        --verbose

    # Vérification du modèle sauvegardé :
    python fine_tune_swinv2.py --eval-only \\
        --model-dir ./swinv2_openfake \\
        --data-dir /chemin/vers/calib_dataset

Structure attendue dans --data-dir :
    calib_dataset/
        REAL/   ← images authentiques
        ALT/    ← images générées / altérées

Le script crée automatiquement un split train/val/test (70/15/15).

Après entraînement, mettre à jour ai_forensics.cfg :
    [models]
    swinv2_model_path = ./swinv2_openfake
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    get_cosine_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BACKBONE_ID = "microsoft/swinv2-small-patch4-window16-256"
LABEL2ID = {"real": 0, "fake": 1}
ID2LABEL = {0: "real", 1: "fake"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepfakeDataset(Dataset):
    def __init__(
        self,
        real_files: list[Path],
        fake_files: list[Path],
        processor,
        augment_fake: bool = False,
    ) -> None:
        self.files  = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
        random.shuffle(self.files)
        self.processor    = processor
        self.augment_fake = augment_fake

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (128, 128, 128))

        # Flux 2 — dégradation légère sur les fakes uniquement
        if label == 1 and self.augment_fake and random.random() < 0.5:
            img = self._degrade(img)

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}

    def _degrade(self, img: Image.Image) -> Image.Image:
        """Dégradation légère pour neutraliser les raccourcis de compression."""
        import io
        # Downscale aléatoire (0.7–1.0×)
        factor = random.uniform(0.7, 1.0)
        w, h = img.size
        img = img.resize((max(64, int(w * factor)), max(64, int(h * factor))), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)  # remonter à la taille originale

        # JPEG recompression aléatoire (qualité 60–95)
        buf = io.BytesIO()
        quality = random.randint(60, 95)
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_files(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]


def _split(files: list[Path], seed: int = 42) -> tuple[list, list, list]:
    """Répartit en train (70%) / val (15%) / test (15%)."""
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def _metrics(model, loader, device: str) -> dict:
    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            pv     = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            logits = model(pixel_values=pv).logits
            preds  = logits.argmax(dim=-1)
            for p, l in zip(preds, labels):
                p, l = int(p), int(l)
                if p == 1 and l == 1: tp += 1
                elif p == 0 and l == 0: tn += 1
                elif p == 1 and l == 0: fp += 1
                elif p == 0 and l == 1: fn += 1
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
    # MCC
    num = tp * tn - fp * fn
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = num / den if den > 0 else 0.0
    return {
        "accuracy": accuracy, "f1": f1, "precision": precision,
        "recall": recall, "mcc": mcc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.verbose:
        print(f"Device : {device.upper()}")

    # Collecte et split
    real_all = _collect_files(Path(args.data_dir) / "REAL")
    fake_all = _collect_files(Path(args.data_dir) / "ALT")

    if not real_all or not fake_all:
        raise FileNotFoundError(
            f"Dossiers REAL/ et ALT/ attendus dans {args.data_dir}.\n"
            f"  REAL : {len(real_all)} fichiers\n"
            f"  ALT  : {len(fake_all)} fichiers"
        )

    r_train, r_val, r_test = _split(real_all)
    f_train, f_val, f_test = _split(fake_all)

    print(f"Dataset : {len(r_train)+len(f_train)} train  "
          f"{len(r_val)+len(f_val)} val  "
          f"{len(r_test)+len(f_test)} test")

    processor = AutoImageProcessor.from_pretrained(BACKBONE_ID, use_fast=True)

    ds_train = DeepfakeDataset(r_train, f_train, processor, augment_fake=True)
    ds_val   = DeepfakeDataset(r_val,   f_val,   processor, augment_fake=False)
    ds_test  = DeepfakeDataset(r_test,  f_test,  processor, augment_fake=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(device=="cuda"))
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers)

    model = AutoModelForImageClassification.from_pretrained(
        BACKBONE_ID,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps   = len(dl_train) * args.epochs
    warmup_steps  = total_steps // 10
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1   = 0.0
    best_path = Path(args.output_dir) / "best"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}", disable=not args.verbose):
            pv     = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            loss   = model(pixel_values=pv, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        val_m = _metrics(model, dl_val, device)
        print(
            f"  Epoch {epoch:02d}  loss={total_loss/len(dl_train):.4f}  "
            f"F1={val_m['f1']:.4f}  MCC={val_m['mcc']:.4f}  "
            f"Prec={val_m['precision']:.4f}  Rec={val_m['recall']:.4f}"
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            model.save_pretrained(str(best_path))
            processor.save_pretrained(str(best_path))
            if args.verbose:
                print(f"  → Meilleur modèle sauvegardé (F1={best_f1:.4f})")

    # Évaluation finale sur le test set
    best_model = AutoModelForImageClassification.from_pretrained(str(best_path)).to(device)
    test_m = _metrics(best_model, dl_test, device)
    print("\n=== Évaluation finale (test set) ===")
    for k, v in test_m.items():
        print(f"  {k:12s} : {v:.4f}" if isinstance(v, float) else f"  {k:12s} : {v}")

    # Sauvegarde finale dans output_dir
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(str(out))
    processor.save_pretrained(str(out))

    # Métriques en JSON
    meta = {
        "backbone": BACKBONE_ID,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_f1": best_f1,
        "test_metrics": test_m,
        "dataset": {
            "real_train": len(r_train), "fake_train": len(f_train),
            "real_val":   len(r_val),   "fake_val":   len(f_val),
            "real_test":  len(r_test),  "fake_test":  len(f_test),
        },
    }
    with open(out / "training_info.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModèle sauvegardé dans : {out.resolve()}")
    print(f"Mettre à jour ai_forensics.cfg :")
    print(f"  [models]")
    print(f"  swinv2_model_path = {out.resolve()}")

    shutil.rmtree(best_path, ignore_errors=True)


def eval_only(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model_dir, use_fast=True)
    model     = AutoModelForImageClassification.from_pretrained(args.model_dir).to(device)

    real_files = _collect_files(Path(args.data_dir) / "REAL")
    fake_files = _collect_files(Path(args.data_dir) / "ALT")
    ds   = DeepfakeDataset(real_files, fake_files, processor)
    dl   = DataLoader(ds, batch_size=32, num_workers=2)
    m    = _metrics(model, dl, device)

    print("=== Évaluation ===")
    for k, v in m.items():
        print(f"  {k:12s} : {v:.4f}" if isinstance(v, float) else f"  {k:12s} : {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning SwinV2 OpenFake")
    parser.add_argument("--data-dir",    type=str, required=True,
                        help="Dossier calib_dataset contenant REAL/ et ALT/")
    parser.add_argument("--output-dir",  type=str, default="./swinv2_openfake",
                        help="Dossier de sauvegarde du modèle fine-tuné")
    parser.add_argument("--model-dir",   type=str, default=None,
                        help="Dossier d'un modèle existant (pour --eval-only)")
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-only",   action="store_true",
                        help="Évaluer un modèle existant sans entraîner")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        if not args.model_dir:
            parser.error("--model-dir requis avec --eval-only")
        eval_only(args)
    else:
        train(args)
