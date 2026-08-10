#!/usr/bin/env python3
"""
Script to prepare NSD stimulus images at 224x224.
"""

import numpy as np
import h5py
from skimage.transform import resize
import os


def load_from_hdf5(hdf5_path):
    """
    Load NSD stimuli images from HDF5 file.

    Args:
        hdf5_path: path to nsd_stimuli.hdf5

    Returns:
        numpy array of shape (N, H, W, 3), dtype uint8
    """
    print(f"Loading images from {hdf5_path}...")
    with h5py.File(hdf5_path, "r") as f:
        images = f["imgBrick"][:]
    print(f"  Loaded shape: {images.shape}, dtype: {images.dtype}")
    return images


def resize_images(images, target_size):
    """
    Resize a batch of images to target size.

    Args:
        images: numpy array of shape (N, H, W, 3) in uint8
        target_size: tuple (height, width) for target resolution

    Returns:
        numpy array of resized images in uint8
    """
    n_images = images.shape[0]
    resized = np.zeros((n_images, target_size[0], target_size[1], 3), dtype=np.uint8)

    print(f"Resizing {n_images} images to {target_size}...")
    for i in range(n_images):
        if i % 1000 == 0:
            print(f"  Processing image {i}/{n_images}")
        resized_img = resize(images[i], target_size, preserve_range=True, anti_aliasing=True)
        resized[i] = resized_img.astype(np.uint8)

    return resized


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'nsd_data')
    imgs_dir = os.path.join(data_dir, 'imgs')

    nsd_images = load_from_hdf5(os.path.join(imgs_dir, 'nsd_stimuli.hdf5'))
    images_224 = resize_images(nsd_images, (224, 224))
    out_path = os.path.join(data_dir, 'nsd_images_224.npy')
    print(f"Saving {out_path}...")
    np.save(out_path, images_224)
    print(f"  Saved shape: {images_224.shape}, dtype: {images_224.dtype}")

    print("\nDone!")


if __name__ == "__main__":
    main()
