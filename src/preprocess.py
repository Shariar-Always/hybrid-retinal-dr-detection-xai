import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


DR_CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


def crop_fundus_borders(img_rgb: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Removes uninformative black borders surrounding circular retinal fundus scans
    so the CNN focuses on the retinal field rather than empty camera margins.
    """
    if img_rgb.ndim == 2:
        mask = img_rgb > tol
        if not np.any(mask):
            return img_rgb
        return img_rgb[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    check_shape = img_rgb[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:
        return img_rgb

    img1 = img_rgb[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = img_rgb[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = img_rgb[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    return np.stack([img1, img2, img3], axis=-1)


def apply_ben_graham_preprocessing(
    image,
    img_size: int = 224,
    sigma_x: int = 10,
) -> Image.Image:
    """
    Applies Ben Graham's retinal fundus normalization:
    1. Crops dark camera margins.
    2. Resizes to (img_size, img_size).
    3. Subtracts local Gaussian-blurred background illumination to enhance
       microaneurysms, hemorrhages, and hard exudates across varied fundus cameras.
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = image

    cropped = crop_fundus_borders(img_rgb)
    resized = cv2.resize(cropped, (img_size, img_size), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX=sigma_x)
    enhanced = cv2.addWeighted(resized, 4, blurred, -4, 128)
    return Image.fromarray(enhanced)


def apply_clahe_enhancement(image, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> Image.Image:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
    to gently enhance retinal vascular contrast without distorting fundus hues.
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = image

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def generate_aptos2019_metadata(
    aptos_dir: str = None,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
) -> bool:
    """
    Builds a stratified, leak-safe metadata.csv from the Kaggle APTOS 2019
    Blindness Detection dataset (train.csv + train_images/).

    Dataset URL:
        https://www.kaggle.com/competitions/aptos2019-blindness-detection
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    candidate_dirs = []
    if aptos_dir:
        candidate_dirs.append(Path(aptos_dir))
    candidate_dirs.extend([
        data_dir / "aptos2019",
        Path("/content/aptos2019"),
        Path("/kaggle/input/aptos2019-blindness-detection"),
        Path(r"E:\Download\aptos2019-blindness-detection"),
    ])

    resolved_dir = None
    for d in candidate_dirs:
        if (d / "train.csv").exists():
            resolved_dir = d
            break

    print("=" * 68)
    print("  APTOS 2019 Retinal Fundus DR Metadata Generator (Stratified)")
    print("=" * 68)

    if resolved_dir is None:
        print("[-] Could not locate APTOS 2019 'train.csv' in any expected directory:")
        for d in candidate_dirs:
            print(f"    - {d}")
        print(
            "\n[i] Download the dataset from:\n"
            "    https://www.kaggle.com/competitions/aptos2019-blindness-detection\n"
            "    Then run: python src/preprocess.py --aptos-dir <path_to_aptos2019>"
        )
        return False

    train_csv_path = resolved_dir / "train.csv"
    print(f"[+] Reading APTOS 2019 labels from: {train_csv_path}")
    df = pd.read_csv(train_csv_path)

    if not {"id_code", "diagnosis"}.issubset(df.columns):
        raise ValueError("Expected columns 'id_code' and 'diagnosis' in APTOS 2019 train.csv")

    df = df.dropna(subset=["id_code", "diagnosis"]).copy()
    df["label"] = df["diagnosis"].astype(int)
    df = df[df["label"].isin(DR_CLASS_NAMES.keys())].copy()
    df["class_name"] = df["label"].map(DR_CLASS_NAMES)
    df["image_path"] = "train_images/" + df["id_code"].astype(str) + ".png"
    df["source"] = "aptos2019"

    # Stratified split: Train (80%) / Val (10%) / Test (10%)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df["label"],
        random_state=seed,
    )
    relative_test_size = test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df = final_df[["image_path", "label", "class_name", "split", "source"]]

    os.makedirs(data_dir, exist_ok=True)
    metadata_csv = data_dir / "metadata.csv"
    final_df.to_csv(metadata_csv, index=False)

    print("-" * 68)
    print(f"[+] Total Retinal Fundus Images Indexed: {len(final_df)}")
    print("\n[+] Split Distribution:")
    print(final_df["split"].value_counts().to_string())
    print("\n[+] ICDR Severity Grade Distribution:")
    for lbl, name in DR_CLASS_NAMES.items():
        cnt = int((final_df["label"] == lbl).sum())
        print(f"    Grade {lbl} ({name:<16}): {cnt}")
    print(f"\n[+] Saved stratified metadata to: {metadata_csv}")
    print("=" * 68)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stratified metadata.csv for APTOS 2019 Diabetic Retinopathy dataset"
    )
    parser.add_argument(
        "--aptos-dir",
        type=str,
        default=None,
        help="Path to extracted APTOS 2019 folder containing train.csv and train_images/",
    )
    parser.add_argument("--val-size", type=float, default=0.10, help="Validation split ratio")
    parser.add_argument("--test-size", type=float, default=0.10, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified split")
    args = parser.parse_args()

    generate_aptos2019_metadata(
        aptos_dir=args.aptos_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
