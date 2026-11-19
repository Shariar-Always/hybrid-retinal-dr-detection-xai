import time
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import cohen_kappa_score, f1_score

from src.models.hybrid_model import HybridDRPredictor
from src.preprocess import apply_ben_graham_preprocessing


class RetinalFundusDataset(Dataset):
    """
    PyTorch Dataset for Retinal Fundus Diabetic Retinopathy Grading (APTOS 2019).
    Optionally applies Ben Graham's retinal illumination normalization.
    """
    def __init__(self, df, data_dir, transform=None, use_ben_graham=True, img_size=224):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_ben_graham = use_ben_graham
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.use_ben_graham:
            image = apply_ben_graham_preprocessing(image, img_size=self.img_size)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    acc = float(np.mean(all_preds_np == all_labels_np))
    qwk = float(cohen_kappa_score(all_labels_np, all_preds_np, weights="quadratic"))
    macro_f1 = float(f1_score(all_labels_np, all_preds_np, average="macro", zero_division=0))

    return total_loss / len(dataloader.dataset), acc, qwk, macro_f1


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid CNN-CBAM Model for 5-Stage Diabetic Retinopathy Grading"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b0",
        choices=["densenet121", "resnet50", "efficientnet_b0"],
        help="Select Hybrid CNN-CBAM backbone",
    )
    parser.add_argument(
        "--no-ben-graham",
        action="store_true",
        help="Disable Ben Graham fundus illumination normalization",
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input spatial resolution")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to APTOS 2019 image root")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to data/metadata.csv")
    args = parser.parse_args()

    use_ben_graham = not args.no_ben_graham

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using Device: {device}")

    metadata_path = (
        Path(args.metadata_path)
        if args.metadata_path
        else (Path("data/metadata.csv") if Path("data/metadata.csv").exists() else Path("../data/metadata.csv"))
    )
    if not metadata_path.exists():
        print(f"[-] Error: Metadata CSV not found at {metadata_path}. Run src/preprocess.py first.")
        return

    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif Path("/content/aptos2019").exists():
        data_dir = Path("/content/aptos2019")
    elif Path("/kaggle/input/aptos2019-blindness-detection").exists():
        data_dir = Path("/kaggle/input/aptos2019-blindness-detection")
    else:
        data_dir = Path("data/aptos2019")

    print(f"[+] Loading metadata from: {metadata_path}")
    print(f"[+] Loading retinal fundus images from: {data_dir}")

    df = pd.read_csv(metadata_path)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    print(f"[+] Dataset split: Train={len(train_df)}, Val={len(val_df)}")

    # Compute inverse-frequency class weights for imbalanced 5-class DR grades
    num_classes = 5
    class_counts = np.array(
        [(train_df["label"] == c).sum() for c in range(num_classes)], dtype=np.float32
    )
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = class_counts.sum() / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[+] Balanced Class Weights (Grades 0-4): {class_weights.cpu().numpy().round(3)}")

    # Rotation-invariant augmentations tailored for retinal fundus photographs
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RetinalFundusDataset(
        train_df, data_dir, transform=train_transform, use_ben_graham=use_ben_graham, img_size=args.img_size
    )
    val_dataset = RetinalFundusDataset(
        val_df, data_dir, transform=val_transform, use_ben_graham=use_ben_graham, img_size=args.img_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"[+] Initializing HybridDRPredictor [Hybrid CNN + CBAM] with {args.backbone} backbone...")
    model = HybridDRPredictor(
        backbone=args.backbone, num_classes=num_classes, pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    weights_name = f"best_dr_model_{args.backbone}.pth"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_save_path = models_dir / weights_name

    best_val_qwk = -1.0
    patience_counter = 0
    patience = 5

    print(f"[+] Starting training for {args.epochs} epochs (tracking Best Validation QWK)...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_qwk, val_f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_qwk)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{args.epochs} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | Val QWK: {val_qwk:.4f} | Macro-F1: {val_f1:.4f}"
        )

        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"[+] Saved best checkpoint (QWK={best_val_qwk:.4f}) -> {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[+] Early stopping triggered at epoch {epoch}.")
                break

    print(f"[+] Training complete! Best Validation QWK: {best_val_qwk:.4f}")


if __name__ == "__main__":
    main()
