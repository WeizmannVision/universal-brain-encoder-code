"""ROI helpers for Algonauts challenge-space masks.

Subject IDs are 1-based (1 = subj01, ..., 8 = subj08).
Masks are expected under: data/nsd_data/roi_masks/subjXX/roi_masks/
"""

import numpy as np

ROI_MASKS_DIR = "data/nsd_data/roi_masks/"

ROI_CLASSES = {
    "prf-visualrois": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"],
    "floc-bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
    "floc-faces": ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
    "floc-places": ["OPA", "PPA", "RSC"],
    "floc-words": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
    "streams": ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"],
}
ROI_NAME_TO_CLASS = {roi: cls for cls, rois in ROI_CLASSES.items() for roi in rois}


def resolve_region(region):
    """Return list of ROI names, or None for full brain."""
    if region in ("full", "all", "*"):
        return None
    if region in ROI_CLASSES:
        return list(ROI_CLASSES[region])
    if region in ROI_NAME_TO_CLASS:
        return [region]
    raise ValueError(f"Unknown region={region!r}")


def get_roi_indices(subject, roi_name, hemisphere="both", roi_masks_dir=ROI_MASKS_DIR):
    """Subject-local voxel indices for one ROI (LH then RH concatenation)."""
    if roi_name not in ROI_NAME_TO_CLASS:
        raise ValueError(f"Unknown roi_name={roi_name!r}")

    roi_class = ROI_NAME_TO_CLASS[roi_name]
    roi_folder = f"{roi_masks_dir}/subj{subject:02d}/roi_masks"
    mapping = np.load(f"{roi_folder}/mapping_{roi_class}.npy", allow_pickle=True).item()
    roi_label = next(k for k, v in mapping.items() if v == roi_name)

    mask_lh = np.load(f"{roi_folder}/lh.{roi_class}_challenge_space.npy")
    mask_rh = np.load(f"{roi_folder}/rh.{roi_class}_challenge_space.npy")
    lh_idx = np.where(mask_lh == roi_label)[0]
    rh_idx = np.where(mask_rh == roi_label)[0]
    lh_len = mask_lh.shape[0]

    if hemisphere == "lh":
        return lh_idx.astype(np.int64)
    if hemisphere == "rh":
        return (rh_idx + lh_len).astype(np.int64)
    if hemisphere == "both":
        return np.concatenate([lh_idx, rh_idx + lh_len]).astype(np.int64)
    raise ValueError(f"hemisphere must be lh, rh, or both, got {hemisphere!r}")


def get_region_indices(subject, region, num_voxels, hemisphere="both", roi_masks_dir=ROI_MASKS_DIR):
    """Subject-local voxel indices for a region ('full', ROI class, or ROI name)."""
    roi_names = resolve_region(region)
    if roi_names is None:
        return np.arange(num_voxels, dtype=np.int64)

    all_idx = []
    for roi in roi_names:
        idx = get_roi_indices(subject, roi, hemisphere=hemisphere, roi_masks_dir=roi_masks_dir)
        idx = idx[(idx >= 0) & (idx < num_voxels)]
        if idx.size > 0:
            all_idx.append(idx)

    if not all_idx:
        raise ValueError(f"No voxels found for region={region!r}, subject={subject}")

    idx = np.concatenate(all_idx)
    _, first = np.unique(idx, return_index=True)
    return idx[np.sort(first)]
