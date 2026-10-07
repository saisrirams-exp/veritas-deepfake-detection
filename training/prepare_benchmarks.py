"""
Benchmark Dataset Preparation Script
Downloads and extracts real/fake splits for:
- Celeb-DF (Cross-Dataset Out-of-Domain Generalization Test)
- FaceForensics++ (Google DFD / Manipulated Benchmark)
"""

import sys
import io
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download


def prepare_celeb_df(output_base: str = "data/celeb_df", max_samples: int = 1000):
    """Extracts balanced real and fake images from Celeb-DF parquet dataset."""
    out_dir = Path(output_base)
    real_dir = out_dir / "real"
    fake_dir = out_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Preparing Celeb-DF Benchmark Split in {out_dir}...")
    try:
        val_path = hf_hub_download(
            repo_id="thenewsupercell/celeb-df-image-dataset",
            filename="data/validation-00000-of-00001.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(val_path)
        print(f"[+] Loaded Celeb-DF validation parquet: {len(df)} total frames.")

        real_count = 0
        fake_count = 0
        target_fake = min(max_samples // 2, 500)
        target_real = min(max_samples // 2, 500)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Celeb-DF"):
            label = int(row["label"])
            img_dict = row["image"]
            img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) and "bytes" in img_dict else None

            if img_bytes is None:
                continue

            if label == 1 and real_count < target_real:
                img_path = real_dir / f"celeb_real_{real_count:05d}.jpg"
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                real_count += 1
            elif label == 0 and fake_count < target_fake:
                img_path = fake_dir / f"celeb_fake_{fake_count:05d}.jpg"
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                fake_count += 1

            if real_count >= target_real and fake_count >= target_fake:
                break

        print(f"[+] Celeb-DF ready: {real_count} real, {fake_count} fake.")

    except Exception as e:
        print(f"[!] Celeb-DF extraction error: {e}", file=sys.stderr)


def prepare_faceforensics(output_base: str = "data/faceforensics", max_frames: int = 400):
    """Downloads FaceForensics++ / Google DFD frames."""
    out_dir = Path(output_base)
    real_dir = out_dir / "real"
    fake_dir = out_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Preparing FaceForensics++ Benchmark Split in {out_dir}...")
    try:
        import urllib.request, json
        req = urllib.request.Request(
            "https://huggingface.co/api/datasets/SUJAY2000/faceforensics-images",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        res = json.loads(urllib.request.urlopen(req).read())
        filenames = [s["rfilename"] for s in res.get("siblings", []) if s["rfilename"].endswith(".jpg")]

        downloaded = 0
        for rfilename in tqdm(filenames[:max_frames], desc="Downloading FF++ Frames"):
            p = hf_hub_download(
                repo_id="SUJAY2000/faceforensics-images",
                filename=rfilename,
                repo_type="dataset"
            )
            dest = fake_dir / Path(rfilename).name
            if not dest.exists():
                with open(p, "rb") as sf, open(dest, "wb") as df:
                    df.write(sf.read())
            downloaded += 1

        print(f"[+] FaceForensics++ fake frames ready: {downloaded} images in {fake_dir}")

        # Real frames from authentic seeds / validation real
        real_sources = list(Path("data/validation/real").glob("*.jpg")) + list(Path("assets").glob("*authentic*.jpg"))
        for i, src in enumerate(real_sources[:downloaded]):
            dest = real_dir / f"ff_real_{i:04d}.jpg"
            if not dest.exists():
                with open(src, "rb") as sf, open(dest, "wb") as df:
                    df.write(sf.read())
        print(f"[+] FaceForensics++ real frames ready: {len(list(real_dir.glob('*.jpg')))} images in {real_dir}")

    except Exception as e:
        print(f"[!] FaceForensics++ extraction error: {e}", file=sys.stderr)


if __name__ == "__main__":
    prepare_celeb_df()
    prepare_faceforensics()
