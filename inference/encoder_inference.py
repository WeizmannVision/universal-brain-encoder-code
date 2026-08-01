"""
Encoder inference: predict fMRI from images.

Default images are the shared test set (type_sample == 2), aligned with
multi_sub_fmri.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.append(os.getcwd())

DATA_DIR = "data/nsd_data/"
TRANSFER_DATA_DIR = "data/nsd_data/transfer/"
MODEL_DIR = "results/saved_models/"
TRANSFER_MODEL_DIR = "results/saved_models/transfer/"
DEFAULT_OUTPUT_DIR = "results/encoder_predictions/"

INNER_CH = 128
DEFAULT_SUBJECTS = [1, 2, 5, 7]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(float)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(float)


def trans_imgs_shift(img):
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img.astype(float)).float()


def parse_args():
    parser = argparse.ArgumentParser(description="Encoder inference: image -> fMRI")
    parser.add_argument(
        "--image_set",
        type=str,
        choices=["test", "ext", "all"],
        default="test",
        help="image set to predict (default: test/shared/multi_sub images)",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="path to custom images .npy (overrides --image_set)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="path to encoder checkpoint",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="subject numbers (1-8) to save; default: 1 2 5 7",
    )
    parser.add_argument(
        "--transfer",
        action="store_true",
        help="use transfer-learning encoder for a single held-out subject",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="held-out subject number (1-8) for transfer inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to save predictions",
    )
    args = parser.parse_args()

    if args.transfer:
        args.subjects = [args.subject]
        args.model = args.model or (
            TRANSFER_MODEL_DIR + f"encoder_transfer_subj{args.subject}.pth"
        )
    else:
        args.model = args.model or MODEL_DIR + f"encoder_ch{INNER_CH}.pth"

    if args.output_dir is None:
        image_label = "custom" if args.images else args.image_set
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, image_label)

    return args


def load_images(args, fmri_data):
    if args.images is not None:
        images_path = args.images
        print(f"Loading custom images from {images_path}...")
        images = np.load(images_path)
        return images

    if args.image_set == "test":
        type_sample = fmri_data["type_sample"]
        images = np.load(DATA_DIR + "nsd_images_224.npy")
        images = images[type_sample == 2]
        print(f"Loading test/shared images ({images.shape[0]} images)...")
    elif args.image_set == "ext":
        images_path = DATA_DIR + "ext_images_224.npy"
        print(f"Loading external images from {images_path}...")
        images = np.load(images_path)
    elif args.image_set == "all":
        images_path = DATA_DIR + "nsd_images_224.npy"
        print(f"Loading all NSD images from {images_path}...")
        images = np.load(images_path)
    else:
        raise ValueError(f"Unknown image set: {args.image_set}")

    return images


def predict_fmri(encoder_model, images, num_voxels_subjects, subjects, device):
    num_images = images.shape[0]
    end = np.cumsum(num_voxels_subjects)
    start = end - num_voxels_subjects

    fmri = {}
    for sub in subjects:
        idx = 0 if len(num_voxels_subjects) == 1 else sub - 1
        n_vox = int(end[idx] - start[idx])
        vox_ind = torch.arange(start[idx], end[idx], device=device).unsqueeze(0)
        sub_fmri = np.zeros([num_images, n_vox], dtype=np.float32)

        print(f"Predicting fMRI for subject {sub} ({num_images} images, {n_vox} voxels)...")
        for i in range(num_images):
            if i % 100 == 0:
                print(f"  Processing image {i}/{num_images}")

            image_tensor = trans_imgs_shift(images[i] / 255.0).unsqueeze(0)
            with torch.no_grad():
                pred = encoder_model(image_tensor.to(device), vox_ind)
            sub_fmri[i] = pred.detach().cpu().numpy()

        fmri[f"subject_{sub}"] = sub_fmri

    return fmri


def save_predictions(fmri, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_dict = {k: v.astype(np.float16) for k, v in fmri.items()}
    output_path = output_dir / "pred_fmri.npz"
    np.savez(output_path, **pred_dict)
    for k, v in pred_dict.items():
        print(f"  {k}: {v.shape}")
    print(f"Saved predictions -> {output_path}")


def compute_voxel_correlations(y_true, y_pred):
    num_voxels = y_true.shape[1]
    corr = np.zeros(num_voxels, dtype=np.float32)
    for i in range(num_voxels):
        corr[i] = pearsonr(y_true[:, i], y_pred[:, i])[0]
    return np.nan_to_num(corr)


def save_correlations(y_true, fmri, num_voxels_subjects, subjects, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    end = np.cumsum(num_voxels_subjects)
    start = end - num_voxels_subjects

    corr_dict = {}
    for sub in subjects:
        idx = sub - 1
        key = f"subject_{sub}"
        sub_corr = compute_voxel_correlations(
            y_true[:, start[idx]:end[idx]],
            fmri[key],
        )
        corr_dict[key] = sub_corr
        print(f"  subject {sub}: median corr {np.median(sub_corr):.4f}")

    output_path = output_dir / "voxel_corr.npz"
    np.savez(output_path, **corr_dict)
    all_corr = np.concatenate(list(corr_dict.values()))
    print(f"Median voxel correlation: {np.median(all_corr):.4f} -> {output_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.transfer:
        fmri_data = np.load(TRANSFER_DATA_DIR + "subjects_single_ses_fmri.npz")
        transfer_sub = f"subj{args.subject}"
        num_voxels_subjects = np.array([fmri_data[transfer_sub].shape[1]])
    else:
        fmri_data = np.load(DATA_DIR + "fmri_v2.npz")
        num_voxels_subjects = fmri_data["num_voxels_subjects"].astype(int)

    images = load_images(args, fmri_data)

    print(f"Loading encoder model from {args.model}...")
    torch.hub.set_dir("data/external_models/torch_hub/")
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")  # needed so encoder model can load
    encoder_model = torch.load(args.model).eval().to(device)
    fmri = predict_fmri(encoder_model, images, num_voxels_subjects, args.subjects, device)
    save_predictions(fmri, args.output_dir)

    y_true = fmri_data["multi_sub_fmri"].astype(np.float32)
    print("Computing per-voxel prediction correlations...")
    save_correlations(y_true, fmri, num_voxels_subjects, args.subjects, args.output_dir)

    print("Encoder inference complete!")


if __name__ == "__main__":
    main()
