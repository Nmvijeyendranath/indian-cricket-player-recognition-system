"""
Experimental CNN Comparison: EfficientNet-B0 Transfer Learning.
Trains a deep CNN baseline on the same leakage-safe cricket player dataset on GPU.
Compares performance, latency, and sample-efficiency against ArcFace + SVM.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CricketFaceDataset(Dataset):
    def __init__(self, manifest_csv: str, root_dir: str, transform=None, label_to_idx=None):
        self.df = pd.read_csv(manifest_csv)
        self.root_dir = Path(root_dir)
        self.transform = transform

        if label_to_idx is None:
            unique_labels = sorted(self.df['identity'].unique())
            self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        # Filter records to only those with valid labels
        self.df = self.df[self.df['identity'].isin(self.label_to_idx)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['path']
        try:
            with Image.open(img_path) as im:
                image = im.convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        label = self.label_to_idx[row['identity']]
        return image, label


def train_efficientnet(
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    root_dir: str,
    models_dir: str = 'models',
    reports_dir: str = 'reports',
    epochs_head: int = 4,
    epochs_fine: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=====================================================================")
    print(f"  EXPERIMENTAL CNN: EFFICIENTNET-B0 TRANSFER LEARNING ON {device.type.upper()}")
    print(f"=====================================================================")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    reports_p = Path(reports_dir)
    reports_p.mkdir(parents=True, exist_ok=True)
    models_p = Path(models_dir)
    models_p.mkdir(parents=True, exist_ok=True)

    # 1. Transforms
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Datasets & Loaders
    train_ds = CricketFaceDataset(train_manifest, root_dir, transform=train_transform)
    label_to_idx = train_ds.label_to_idx
    num_classes = len(label_to_idx)
    print(f"Total eligible classes: {num_classes} | Train samples: {len(train_ds)}")

    val_ds = CricketFaceDataset(val_manifest, root_dir, transform=val_transform, label_to_idx=label_to_idx)
    test_ds = CricketFaceDataset(test_manifest, root_dir, transform=val_transform, label_to_idx=label_to_idx)
    print(f"Val samples: {len(val_ds)} | Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 3. Model Architecture
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_weights_path = models_p / 'efficientnet_b0_best.pt'

    # PHASE 1: Train classification head with frozen backbone
    print("\n--- Phase 1: Training Classification Head (Backbone Frozen) ---")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs_head + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            total_train += len(targets)

        # Validation
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += len(targets)

        tr_loss = train_loss / total_train
        tr_acc = train_correct / total_train
        v_loss = val_loss / total_val
        v_acc = val_correct / total_val

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)

        print(f"Epoch {epoch}/{epochs_head} [{time.time()-t0:.1f}s] - Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_weights_path)

    # PHASE 2: Unfreeze top layers and fine-tune
    print("\n--- Phase 2: Fine-Tuning Top Convolutional Layers ---")
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    optimizer_fine = torch.optim.AdamW([
        {'params': model.features[-2:].parameters(), 'lr': lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': lr * 0.5}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fine, T_max=epochs_fine)

    for epoch in range(1, epochs_fine + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer_fine.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_fine.step()

            train_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            total_train += len(targets)

        scheduler.step()

        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += len(targets)

        tr_loss = train_loss / total_train
        tr_acc = train_correct / total_train
        v_loss = val_loss / total_val
        v_acc = val_correct / total_val

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)

        print(f"Fine-tune Epoch {epoch}/{epochs_fine} [{time.time()-t0:.1f}s] - Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}% | Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%")
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_weights_path)

    # 4. Evaluate Best EfficientNet on Test Set
    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()

    test_correct, test_top3, total_test = 0, 0, 0
    t_inf_start = time.time()
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == targets).sum().item()

            # Top-3
            _, top3_indices = outputs.topk(min(3, num_classes), dim=1)
            for i in range(len(targets)):
                if targets[i] in top3_indices[i]:
                    test_top3 += 1
            total_test += len(targets)

    inf_time_total = time.time() - t_inf_start
    test_top1_acc = test_correct / total_test
    test_top3_acc = test_top3 / total_test
    ms_per_img = (inf_time_total / total_test) * 1000

    print(f"\n=====================================================================")
    print(f"  EFFICIENTNET-B0 TEST RESULTS (Held-Out Test Set N={total_test})")
    print(f"  Top-1 Accuracy: {test_top1_acc * 100:.2f}%")
    print(f"  Top-3 Accuracy: {test_top3_acc * 100:.2f}%")
    print(f"  Inference Latency: {ms_per_img:.2f} ms/image on {device.type.upper()}")
    print(f"=====================================================================")

    # 5. Plot Loss & Accuracy Curves
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='orange')
    plt.axvline(epochs_head, color='gray', linestyle='--', label='Unfreeze Backbone')
    plt.title('EfficientNet-B0 Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [a * 100 for a in history['train_acc']], label='Train Acc', color='blue')
    plt.plot(epochs_range, [a * 100 for a in history['val_acc']], label='Val Acc', color='orange')
    plt.axvline(epochs_head, color='gray', linestyle='--', label='Unfreeze Backbone')
    plt.title('EfficientNet-B0 Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_p / 'efficientnet_training_curves.png', dpi=200)
    plt.close()

    # Save summary metrics
    eff_summary = {
        'model': 'EfficientNet-B0',
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'test_samples': total_test,
        'num_classes': num_classes,
        'test_top1_accuracy': float(test_top1_acc),
        'test_top3_accuracy': float(test_top3_acc),
        'inference_ms_per_image': float(ms_per_img),
        'device': str(device)
    }
    with open(reports_p / 'efficientnet_summary.json', 'w', encoding='utf-8') as f:
        json.dump(eff_summary, f, indent=2)

    return eff_summary


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 baseline.")
    parser.add_argument('--train-manifest', default='data/manifest_train.csv')
    parser.add_argument('--val-manifest', default='data/manifest_val.csv')
    parser.add_argument('--test-manifest', default='data/manifest_test.csv')
    parser.add_argument('--root', default='data/archive (3)/indian cricketer')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--reports-dir', default='reports')
    parser.add_argument('--epochs-head', type=int, default=4)
    parser.add_argument('--epochs-fine', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_efficientnet(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        root_dir=args.root,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        epochs_head=args.epochs_head,
        epochs_fine=args.epochs_fine,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == '__main__':
    main()

