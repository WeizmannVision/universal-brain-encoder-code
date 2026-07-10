from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.io import loadmat

# ── Paths (relative to this script) ───────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NSD_DATA = SCRIPT_DIR.parent / "nsd_data"

RAW_FMRI_DIR   = str(NSD_DATA / "raw_fmri")
ROI_MASKS_DIR  = str(NSD_DATA / "roi_masks")
EXPERIMENT_DIR = str(NSD_DATA / "experiment")
OUT_DIR        = str(NSD_DATA)

# number of sessions per subject (subjects 1-8)
sub_ses = [40, 40, 32, 30, 40, 32, 40, 30]

# ── Load and normalise per-subject fMRI betas ─────────────────────────────────
print("Loading per-subject fMRI betas...")
sub_fmri = {}
for s in range(1, 9):
    sub = str(s)
    dir_ = RAW_FMRI_DIR + "/subj0" + sub
    num_ses = sub_ses[s - 1]
    print(f"  subject {s}/8: loading {num_ses} sessions")

    space_lh = np.load(ROI_MASKS_DIR + "/sub" + sub + "/lh.all-vertices_fsaverage_space.npy")
    space_rh = np.load(ROI_MASKS_DIR + "/sub" + sub + "/rh.all-vertices_fsaverage_space.npy")

    Y_sub = []
    for i in range(1, num_ses + 1):
        if i % 10 == 0:
            print(f"    session {i}/{num_ses}")
        ses = str(i)
        if len(ses) < 2:
            ses = "0" + ses

        file = nib.load(dir_ + "/lh.betas_session" + ses + ".mgh")
        lh = file.get_fdata()[:, 0, 0, :].transpose()
        lh = lh[:, space_lh.astype(bool)]

        file = nib.load(dir_ + "/rh.betas_session" + ses + ".mgh")
        rh = file.get_fdata()[:, 0, 0, :].transpose()
        rh = rh[:, space_rh.astype(bool)]

        # concatenate hemispheres and z-score across voxels
        Y_i = np.concatenate([lh, rh], axis=1)
        Y_i = (Y_i - Y_i.mean(axis=0, keepdims=True)) / Y_i.std(axis=0, keepdims=True)
        Y_i = np.nan_to_num(Y_i)
        Y_sub.append(Y_i)

    sub_fmri[s] = np.concatenate(Y_sub)
    print(f"  subject {s}/8 done: shape {sub_fmri[s].shape}")

# voxel counts per subject, derived from loaded data
num_voxels_subjects = np.array([sub_fmri[s].shape[1] for s in range(1, 9)])
print(f"Voxel counts per subject: {num_voxels_subjects}")

# ── Experiment design ─────────────────────────────────────────────────────────
print("Loading experiment design...")
mat = loadmat(EXPERIMENT_DIR + "/nsd_expdesign.mat")

# maps trial index -> global image index (0-based)
ses_to_sub = mat['masterordering'][0, :] - 1
# maps (subject, repeat) -> global image index (0-based)
sub_to_global = mat['subjectim'] - 1

# ── Build fMRI arrays indexed by global image ──────────────────────────────────
print("Building fMRI arrays indexed by global image...")
NUM_IMAGES = 73000
NUM_VOXELS = int(num_voxels_subjects[0])

vox_sub    = num_voxels_subjects.astype(int)
vox_cumsum = np.cumsum(vox_sub)
last_ind   = vox_cumsum
first_ind  = vox_cumsum - vox_sub

# first 1000 images seen by all subjects (multi-subject images)
mult_inds = sub_to_global[0][:1000]

# (image, repeat, voxel) — up to 3 repeats; padded with NaN
single_sub_fmri = np.full((NUM_IMAGES, 3, NUM_VOXELS),              np.nan, dtype=float)
multi_sub_fmri  = np.full((1000,       3, int(num_voxels_subjects.sum())), np.nan, dtype=float)

single_sub_num_repeats = np.zeros(NUM_IMAGES)
multi_sub_num_repeats  = np.zeros([1000, 8])

for subj in range(1, 9):
    fmri = sub_fmri[subj]
    print(f"  indexing subject {subj}/8: {fmri.shape[0]} trials")
    for i in range(fmri.shape[0]):
        global_ind = sub_to_global[subj-1,ses_to_sub[i]]
        repeat = min(int(single_sub_num_repeats[global_ind]),2)
        single_sub_fmri[global_ind,repeat,:vox_sub[subj-1]] = fmri[i]
        first_ind_i = first_ind[subj-1]
        last_ind_i = last_ind[subj-1]
        if(global_ind in mult_inds):
            mult_ind = np.where(mult_inds==global_ind)[0][0]
            repeat_mult = int(multi_sub_num_repeats[mult_ind,subj-1])
            multi_sub_fmri[mult_ind,repeat_mult,first_ind_i:last_ind_i] =  fmri[i]
            multi_sub_num_repeats[mult_ind,subj-1]+=1
        
        single_sub_num_repeats[global_ind] += 1

# ── Per-image subject assignment ───────────────────────────────────────────────
print("Assigning subjects per image...")
single_sub = np.zeros(NUM_IMAGES)
for subj in range(1, 9):
    fmri = sub_fmri[subj]
    for i in range(fmri.shape[0]):
        global_ind = sub_to_global[subj - 1, ses_to_sub[i]]
        single_sub[global_ind] = subj
single_sub = single_sub.astype(int)

type_desc = np.array({0: 'no measurment', 1: 'single sub', 2: 'multi sub'}, dtype=object)

# ── Sample type: 0=unseen, 1=single-subject, 2=multi-subject ──────────────────
print("Averaging repeats and building validation split...")
type_sample = np.ones([NUM_IMAGES])
type_sample[mult_inds] = 2
type_sample[single_sub_num_repeats == 0] = 0

# average across repeats
single_sub_fmri_mean = np.nanmean(single_sub_fmri, axis=1)[type_sample == 1]
single_sub_fmri_mean = np.nan_to_num(single_sub_fmri_mean, nan=0)
multi_sub_fmri_mean  = np.nanmean(multi_sub_fmri, axis=1)

# ── Validation indices ────────────────────────────────────────────────────────
single_sub_inds = np.where(type_sample == 1)[0]
val_ind = np.random.choice(single_sub_inds, size=2500, replace=False)

val_inds_onehot = np.zeros(NUM_IMAGES)
val_inds_onehot[val_ind] = 1
val_single_ind = np.where(val_inds_onehot[type_sample == 1])[0]

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = OUT_DIR + "/fmri_v2.npz"
print(f"Saving {out_path}...")
np.savez(
    out_path,
    num_voxels_subjects  = num_voxels_subjects,
    type_sample          = type_sample,
    single_sub_fmri      = single_sub_fmri_mean.astype(np.float16),
    multi_sub_fmri       = multi_sub_fmri_mean.astype(np.float16),
    valid_multi_sub_fmri = multi_sub_num_repeats > 0,
    type_desc            = type_desc,
    single_sub           = single_sub[type_sample == 1],
    val_global_ind       = val_ind,
    val_single_ind       = val_single_ind,
)
print("Done.")
