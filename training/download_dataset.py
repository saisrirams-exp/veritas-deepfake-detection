"""
Dataset Downloader and Extractor for SOTA Deepfake Training
Downloads public datasets from Hugging Face Hub and organizes into train/val splits.
"""

import sys
import io
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download


def download_and_extract(
    repo_id: str = "itsLeen/deepfake_vs_real_image",
    output_base: str = "data",
    download_train: bool = True
):
    out_dir = Path(output_base)
    
    train_real = out_dir / "train" / "real"
    train_fake = out_dir / "train" / "fake"
    val_real = out_dir / "validation" / "real"
    val_fake = out_dir / "validation" / "fake"

    for d in (train_real, train_fake, val_real, val_fake):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading test/validation split from {repo_id}...")
    try:
        test_path = hf_hub_download(
            repo_id=repo_id,
            filename="data/test-00000-of-00001.parquet",
            repo_type="dataset"
        )
        print(f"[+] Downloaded test parquet to: {test_path}")

        df_test = pd.read_parquet(test_path)
        print(f"[*] Extracting {len(df_test)} validation images...")

        for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Saving Validation Images"):
            img_bytes = row["image"]["bytes"]
            label = int(row["label"])  # 1 = Real, 0 = Fake
            target_dir = val_real if label == 1 else val_fake
            img_path = target_dir / f"val_{idx:05d}.jpg"
            
            with open(img_path, "wb") as f:
                f.write(img_bytes)

        print(f"[+] Validation split ready: {len(list(val_real.glob('*.jpg')))} real, {len(list(val_fake.glob('*.jpg')))} fake.")

    except Exception as e:
        print(f"[!] Error downloading test split: {e}", file=sys.stderr)

    if download_train:
        print(f"[*] Downloading training split (434 MB) from {repo_id}...")
        try:
            train_path = hf_hub_download(
                repo_id=repo_id,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset"
            )
            print(f"[+] Downloaded train parquet to: {train_path}")

            df_train = pd.read_parquet(train_path)
            print(f"[*] Extracting {len(df_train)} training images...")

            for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Saving Training Images"):
                img_bytes = row["image"]["bytes"]
                label = int(row["label"])
                target_dir = train_real if label == 1 else train_fake
                img_path = target_dir / f"train_{idx:05d}.jpg"

                with open(img_path, "wb") as f:
                    f.write(img_bytes)

            print(f"[+] Training split ready: {len(list(train_real.glob('*.jpg')))} real, {len(list(train_fake.glob('*.jpg')))} fake.")

        except Exception as e:
            print(f"[!] Error downloading train split: {e}", file=sys.stderr)


if __name__ == "__main__":
    download_and_extract()
