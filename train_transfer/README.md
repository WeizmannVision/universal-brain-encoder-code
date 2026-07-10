# Transfer Learning (Encoder)

This directory contains scripts for adapting a pretrained encoder to a new subject.

> **Note:** This transfer learning procedure is performed within the Natural Scenes Dataset (NSD), using held-out subjects as the transfer target. It can be easily modified to transfer to other datasets; the default NSD encoder checkpoint trained on all subjects can be used as the reference model.

## Overview

The encoder transfer pipeline has two stages:

1. **Base training** — Train an encoder on 7 NSD subjects, holding out one subject (e.g. `--remove_sub 1` excludes subject 1). This follows the same procedure as regular encoder training in the main [README](../README.md), but excludes the held-out subject from training and validation.
2. **Transfer training** — Fine-tune the base encoder on the held-out subject using a small amount of fmri data.

Throughout this README, `N` denotes the held-out subject number (1–8). Use the same value consistently across all steps.

---

## Base Training (7 subjects)

Base training uses the processed NSD data from the main [README](../README.md) (download and data processing steps). No additional data preparation is needed.

```bash
python train_transfer/train_encoder_base.py --remove_sub N
```

---

## Transfer Training (held-out subject)

### Train Encoder (NSD data)

```bash
python train_transfer/train_encoder_transfer.py --SUBJECT N
```

### Train Encoder (custom image/fMRI pairs)

Provide subject-local fMRI of shape `[N, V]` and matching images `[N, H, W, 3]`:

```bash
python train_transfer/train_encoder_transfer_custom.py \
  --images train_images.npy \
  --fmri train_fmri.npy \
  --val_images val_images.npy \
  --val_fmri val_fmri.npy
```

Optional: `--name`, `--base_model path.pth`, `--NUM_SAMPLES K`, `--epochs E`.
If validation files are omitted, the checkpoint from the last epoch is saved.

---

## Inference

```bash
python inference/encoder_inference.py --transfer --subject N
```
