"""
Predict fMRI from an arbitrary image array (.npy).

Optionally restrict predictions to an ROI / ROI family via --region.
For NSD test-set evaluation with correlations, use encoder_inference.py.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.getcwd())

from utils.roi_utils import get_region_indices

DATA_DIR = "data/nsd_data/"
MODEL_DIR = "results/saved_models/"
DEFAULT_OUTPUT_DIR = "results/encoder_predictions/custom/"

INNER_CH = 128
DEFAULT_SUBJECTS = [1, 2, 5, 8]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(float)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(float)


def trans_imgs_shift(img):
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img.astype(float)).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Predict fMRI for custom images")
    parser.add_argument("--images", type=str, required=True, help="path to images .npy [N,H,W,3]")
    parser.add_argument("--model", type=str, default=None, help="encoder checkpoint path")
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="subject numbers (1-8); default: 1 2 5 8",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="full",
        help="full | ROI class (e.g. floc-faces) | ROI name (e.g. EBA)",
    )
    parser.add_argument(
        "--hemisphere",
        type=str,
        choices=["both", "lh", "rh"],
        default="both",
        help="hemisphere for ROI selection (ignored for region=full)",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.model = args.model or MODEL_DIR + f"encoder_ch{INNER_CH}.pth"
    return args


def predict_fmri_for_images(encoder_model, images, num_voxels_subjects, subjects, region, hemisphere, device):
    num_images = images.shape[0]
    end = np.cumsum(num_voxels_subjects)
    start = end - num_voxels_subjects

    fmri = {}
    for sub in subjects:
        idx = sub - 1
        n_vox = int(end[idx] - start[idx])
        local_idx = get_region_indices(sub, region, n_vox, hemisphere=hemisphere)
        vox_ind = torch.from_numpy((start[idx] + local_idx).astype(np.int64)).to(device).unsqueeze(0)
        sub_fmri = np.zeros([num_images, len(local_idx)], dtype=np.float32)

        print(f"Predicting subject {sub}: {num_images} images, {len(local_idx)} voxels (region={region})")
        for i in range(num_images):
            if i % 100 == 0:
                print(f"  Processing image {i}/{num_images}")
            image_tensor = trans_imgs_shift(images[i] / 255.0).unsqueeze(0)
            with torch.no_grad():
                pred = encoder_model(image_tensor.to(device), vox_ind)
            sub_fmri[i] = pred.detach().cpu().numpy()

        fmri[f"subject_{sub}"] = sub_fmri

    return fmri


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fmri_data = np.load(DATA_DIR + "fmri_v2.npz")
    num_voxels_subjects = fmri_data["num_voxels_subjects"].astype(int)

    print(f"Loading images from {args.images}...")
    images = np.load(args.images)

    print(f"Loading encoder model from {args.model}...")
    encoder_model = torch.load(args.model).eval().to(device)

    fmri = predict_fmri_for_images(
        encoder_model, images, num_voxels_subjects, args.subjects, args.region, args.hemisphere, device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pred_fmri.npz"
    np.savez(output_path, **{k: v.astype(np.float16) for k, v in fmri.items()})
    for k, v in fmri.items():
        print(f"  {k}: {v.shape}")
    print(f"Saved predictions -> {output_path}")


if __name__ == "__main__":
    main()
