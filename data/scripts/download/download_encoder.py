#!/usr/bin/env python3
"""Download pretrained encoder checkpoint from Hugging Face.

Source: https://huggingface.co/RomanBeliy/Brain-IT/blob/main/checkpoints/encoder_ch128.pth
Requires: huggingface-hub (included in env.yml)
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "RomanBeliy/Brain-IT"
FILENAME = "checkpoints/encoder_ch128.pth"

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parents[2] / "results" / "saved_models"
OUT_FILE = OUT_DIR / "encoder_ch128.pth"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists():
        print(f"skip (exists): {OUT_FILE}")
        return

    print(f"Downloading {REPO_ID}/{FILENAME} -> {OUT_FILE}")
    cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    shutil.copy2(cached, OUT_FILE)
    print(f"Done. Encoder saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
