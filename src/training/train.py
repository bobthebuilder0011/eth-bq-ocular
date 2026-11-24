import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import BilateralFundusDataset
from models.eth_bq import ETHBQFramework
from models.loss import AsymmetricFocalLoss


def train(args):
    # force CPU - quantum simulator doesn't support MPS/CUDA
    device = torch.device("cpu")
    print(f"Training on: {device}")

    manifest_path = args.manifest or os.path.join(PROJECT_ROOT, "data", "benchmark", "dataset_manifest.json")
    ckpt_dir = args.checkpoint_dir or os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    train_ds = BilateralFundusDataset(manifest["train_samples"], augment=True)
    val_ds   = BilateralFundusDataset(manifest["test_samples"],  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ETHBQFramework(
        backbone_name=args.backbone,
        embed_dim=256,
        num_heads=4,
        n_qubits=8,
        n_quantum_layers=3,
        num_classes=8,
        pretrained=True,
        use_quantum=True,
        use_bilateral_attention=True,
    ).to(device)

    criterion = AsymmetricFocalLoss(gamma_neg=4.0, gamma_pos=0.0, margin=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()

        for batch in train_loader:
            od = batch["od"].to(device)
            os_ = batch["os"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(od, os_)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = logits.argmax(-1)
            ep_loss += loss.item() * od.size(0)
            ep_correct += (preds == labels).sum().item()
            ep_total += od.size(0)

        scheduler.step()
        train_loss = ep_loss / ep_total
        train_acc  = ep_correct / ep_total * 100

        # --- validation ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                od = batch["od"].to(device)
                os_ = batch["os"].to(device)
                labels = batch["label"].to(device)
                logits, _ = model(od, os_)
                loss = criterion(logits, labels)
                preds = logits.argmax(-1)
                v_loss += loss.item() * od.size(0)
                v_correct += (preds == labels).sum().item()
                v_total += od.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total * 100
        elapsed  = time.time() - t0

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 2))

        print(f"Epoch [{epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
              f"({elapsed:.1f}s)")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_acc": val_acc, "train_acc": train_acc},
                os.path.join(ckpt_dir, "eth_bq_best.pth"),
            )

    with open(os.path.join(ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--backbone",      type=str,   default="efficientnet_b0")
    p.add_argument("--manifest",      type=str,   default=None)
    p.add_argument("--checkpoint-dir",type=str,   default=None)
    args = p.parse_args()
    train(args)
