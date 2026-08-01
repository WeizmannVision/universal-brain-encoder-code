"""
Transfer training on custom image/fMRI pairs.

Same freeze + voxel_embed fine-tune as train_encoder_transfer.py, but loads
training (and optional validation) data from user-provided .npy files.

  --images   [N, H, W, 3] uint8
  --fmri     [N, V] float, subject-local voxels
  --val_images / --val_fmri  optional; if omitted, no validation / saves last epoch
"""

import os
import sys
sys.path.append(os.getcwd())
gpu = "0"
dropout = 0.25
embed_dim_vox = 256
inner_ch = 128

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

import torch
import torch.optim as optim
from torch.utils import data
from tensorboardX import SummaryWriter
from scipy.ndimage import shift
import numpy as np
import argparse

from utils.train_utils_enc import train, test
from models.encoder_models import encoder_param
from utils.datasets import EncDataset

torch.set_default_tensor_type("torch.FloatTensor")
device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, required=True, help="train images .npy [N,H,W,3]")
parser.add_argument("--fmri", type=str, required=True, help="train fMRI .npy [N,V] subject-local")
parser.add_argument("--val_images", type=str, default=None, help="optional val images .npy")
parser.add_argument("--val_fmri", type=str, default=None, help="optional val fMRI .npy")
parser.add_argument("--base_model", type=str, default=f"results/saved_models/encoder_ch{inner_ch}.pth", help="base encoder checkpoint")
parser.add_argument("--name", type=str, default="encoder_transfer_custom", help="model name for saving")
parser.add_argument("--NUM_SAMPLES", dest="num_samples", type=int, default=None, help="optional cap on training samples")
parser.add_argument("--epochs", type=int, default=50)
args = parser.parse_args()

batch_size = 32
lr = 1e-3
epochs = args.epochs

print(f"Loading train images from {args.images}")
embeds_single_train = np.load(args.images)
print(f"Loading train fMRI from {args.fmri}")
single_sub_fmri_train = np.load(args.fmri).astype(np.float32)
if args.num_samples is not None:
    embeds_single_train = embeds_single_train[: args.num_samples]
    single_sub_fmri_train = single_sub_fmri_train[: args.num_samples]
single_sub_train = np.zeros(single_sub_fmri_train.shape[0], dtype=int)

num_voxels_subjects = np.array([single_sub_fmri_train.shape[1]])
NUM_VOXELS = int(num_voxels_subjects.sum())

use_val = args.val_images is not None and args.val_fmri is not None
if use_val:
    print(f"Loading val images from {args.val_images}")
    embeds_single_val = np.load(args.val_images)
    print(f"Loading val fMRI from {args.val_fmri}")
    single_sub_fmri_val = np.load(args.val_fmri).astype(np.float32)
    single_sub_val = np.zeros(single_sub_fmri_val.shape[0], dtype=int)

name = args.name
print(name)

enc_param = encoder_param(NUM_VOXELS)
enc_param.inner_ch = inner_ch
enc_param.drop_out = dropout
enc_param.embed_dim_vox = embed_dim_vox
enc_param.in_spatial = 257

dataloader_param = {"batch_size": batch_size, "shuffle": True, "num_workers": 4}
dataloader_val_param = {"batch_size": 8, "shuffle": True, "num_workers": 1}

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(float)
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(float)


def trans_imgs(imgs):
    imgs = imgs / 255.0
    imgs = (imgs - mean) / std
    imgs = imgs.transpose([2, 0, 1])
    return torch.from_numpy(imgs.astype(float)).float()


def rand_shift(img, max_shift=0):
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    return shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode="nearest")


def trans_imgs_shift(img, max_shift=3):
    img = img / 255.0
    img = rand_shift(img, max_shift)
    img = (img - mean) / std
    img = img.transpose([2, 0, 1])
    return torch.from_numpy(img.astype(float)).float()


fmri_dataset_train = EncDataset(
    embeds_single_train,
    single_sub_fmri_train,
    single_sub_train,
    num_voxels_subjects,
    preprocess=trans_imgs_shift,
    num_voxels_to_sample=5000,
)
train_generator = data.DataLoader(fmri_dataset_train, **dataloader_param)

if use_val:
    fmri_dataset_val = EncDataset(
        embeds_single_val,
        single_sub_fmri_val,
        single_sub_val,
        num_voxels_subjects,
        sample=False,
        preprocess=trans_imgs,
    )
    val_generator = data.DataLoader(fmri_dataset_val, **dataloader_val_param)

writer = SummaryWriter("logs/tensorboard/encoder_exp/" + name)


def main():
    print(f"Loading base model from {args.base_model}")
    torch.hub.set_dir("data/external_models/torch_hub/")
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")  # needed so encoder model can load
    model = torch.load(args.base_model)
    for param in model.parameters():
        param.requires_grad = False
    model.voxel_embed = torch.nn.Parameter(
        (enc_param.init / (2 * np.sqrt(enc_param.embed_dim_vox)))
        * torch.randn(NUM_VOXELS, enc_param.embed_dim_vox),
        requires_grad=True,
    ).float()
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    os.makedirs("results/saved_models/transfer/", exist_ok=True)
    save_path = "results/saved_models/transfer/" + name + ".pth"

    best_metric = 0
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(model, device, train_generator, optimizer, epoch, writer)
        if use_val:
            metric = test(model, device, val_generator, epoch, writer)
            if metric > best_metric:
                best_metric = metric
                torch.save(model, save_path)
                print(f"Saved best model (metric={best_metric:.4f})")
        else:
            torch.save(model, save_path)

    if not use_val:
        print(f"Model saved after {epochs} epochs -> {save_path}")


if __name__ == "__main__":
    main()
