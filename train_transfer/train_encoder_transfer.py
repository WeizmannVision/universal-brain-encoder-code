import os
import sys
sys.path.append(os.getcwd())
gpu     = "0"#str(sys.argv[1])
layer_attn_temp_factor   = None#float(sys.argv[2])
r = 7#int(sys.argv[2])
alpha = 0.01#float(sys.argv[3])
model_size = 2 #int(sys.argv[4])

dropout     =  0.25  # float(sys.argv[4])
embed_dim_vox = 256#int(sys.argv[1])
inner_ch = 128 #int(sys.argv[1])


os.environ["CUDA_VISIBLE_DEVICES"] = gpu

select_layers= [1,6,12,18,24]
#select_layers= [1,5,10,15,20,24]   



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from torchvision import datasets, transforms


from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
device = torch.device("cuda")
import numpy as np


import scipy.stats as stat
import sklearn.model_selection as sk_m
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import copy
import sys
import math


import argparse
import numpy as np
from utils.train_utils_enc import train, test
from models.encoder_models import Encoder, encoder_param
from utils.datasets import EncDataset
torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser()
parser.add_argument('--SUBJECT', dest='subject', type=int, default=1, help='subject number (1-8) to fine-tune on')
parser.add_argument('--NUM_SAMPLES', dest='num_samples', type=int, default=1000, help='number of samples to use for transfer')
args = parser.parse_args()

TRANSFER_SUB = f"subj{args.subject}"  # subject held out in base training, now fine-tuned on
batch_size = 32
lr = 1e-3 #0.001
num_samples = args.num_samples
epochs = 50

#save_model = '/home/romanb/data/NSD_models/model.pth'
data_dir      = "data/nsd_data/"
base_save_dir = "results/saved_models/transfer/"

fmri_data = np.load(data_dir + "fmri_v2.npz")
num_voxels_all = fmri_data['num_voxels_subjects'].astype(int)
type_sample = fmri_data["type_sample"]
single_sub = fmri_data['single_sub']
single_sub_fmri = fmri_data['single_sub_fmri']
embeds = np.load(data_dir + "nsd_images_224.npy")
val_ind = fmri_data['val_single_ind']
train_ind = np.ones(single_sub_fmri.shape[0], dtype=bool)
train_ind[val_ind] = False

embeds_single = embeds[type_sample == 1]
idx = args.subject - 1
end = np.cumsum(num_voxels_all)
start = end - num_voxels_all

embeds_single_train = embeds_single[train_ind]
single_sub_fmri_train = single_sub_fmri[train_ind]
single_sub_train = single_sub[train_ind]

sub_mask = single_sub_train == args.subject
embeds_single_train = embeds_single_train[sub_mask][:num_samples]
single_sub_fmri_train = single_sub_fmri_train[sub_mask][:num_samples, start[idx]:end[idx]]
single_sub_train = np.zeros(single_sub_fmri_train.shape[0], dtype=int)

embeds_single_val = embeds_single[val_ind]
single_sub_fmri_val = single_sub_fmri[val_ind]
single_sub_val = single_sub[val_ind]

val_mask = single_sub_val == args.subject
embeds_single_val = embeds_single_val[val_mask]
single_sub_fmri_val = single_sub_fmri_val[val_mask, start[idx]:end[idx]]
single_sub_val = np.zeros(single_sub_fmri_val.shape[0], dtype=int)

num_voxels_subjects = np.array([int(end[idx] - start[idx])])

NUM_VOXELS = int(num_voxels_subjects.sum())


layer_attn_temp = 20#layer_attn_temp
layer_attn_temp_factor = layer_attn_temp_factor#layer_attn_temp_factor

name = f"encoder_transfer_{TRANSFER_SUB}"

print(name)
enc_param = encoder_param(NUM_VOXELS)

enc_param.inner_ch = inner_ch
enc_param.drop_out = dropout

 #1280#
 #256#197
enc_param.embed_dim_vox = embed_dim_vox

dataloader_param = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 4}

dataloader_val_param = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 1}

#enc_param.init = 0.1
###################################################
from scipy.ndimage import shift

mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3]).astype(float)
std  = np.array([0.229, 0.224, 0.225]).reshape([1,1,3]).astype(float) 
def trans_imgs(imgs):
    imgs = imgs/255.0
    imgs = (imgs-mean)/std
    imgs =  imgs.transpose([2,0,1])
    imgs =  torch.from_numpy(imgs.astype(float)).float()
    return imgs


def rand_shift(img,max_shift = 0 ):
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    img_shifted = shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode='nearest')
    return img_shifted

def trans_imgs_shift(img, max_shift = 3):
    img = img/255.0
    img = rand_shift(img,max_shift)
    img = (img-mean)/std
    img =  img.transpose([2,0,1])
    img =  torch.from_numpy(img.astype(float)).float()
    return img


fmri_dataset_train = EncDataset(embeds_single_train, single_sub_fmri_train, single_sub_train, num_voxels_subjects, preprocess = trans_imgs_shift ,num_voxels_to_sample = 5000)
fmri_dataset_val  = EncDataset(embeds_single_val, single_sub_fmri_val, single_sub_val, num_voxels_subjects, sample = False, preprocess = trans_imgs)
train_generator = data.DataLoader(fmri_dataset_train, **dataloader_param)
val_generator = data.DataLoader(fmri_dataset_val, **dataloader_val_param)



writer = SummaryWriter('logs/tensorboard/encoder_exp/'+name)  #
include_reg_tokens = False
if(include_reg_tokens):
    enc_param.in_spatial = 261
else:
    enc_param.in_spatial = 257

#select_layers= [1,3,6,9,12]

torch.hub.set_dir("data/external_models/torch_hub/")
if(model_size == 0):
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    enc_param.in_channels = 384
if(model_size == 1):
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    enc_param.in_channels = 768
elif(model_size == 2):
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    enc_param.in_channels = 1024
    select_layers= [1,6,12,18,24]
    
elif(model_size == 3):
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    enc_param.in_channels = 1536
    #select_layers= [1,10,20,30,40]
    select_layers= [1,5,10,15,20,25,30,35,40]

encoder = encoder.eval().cuda()   

## freeze weights
for param in encoder.parameters():
    param.requires_grad = False
    

    

def main():
    # Load pretrained base model (trained without this subject)
    model = torch.load(base_save_dir + f"encoder_ch{inner_ch}_base_remove_sub_{args.subject}.pth")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Reinitialize voxel_embed as trainable
    model.voxel_embed = torch.nn.Parameter((enc_param.init/(2*np.sqrt(enc_param.embed_dim_vox)))*torch.randn(NUM_VOXELS, enc_param.embed_dim_vox), requires_grad=True).float()
    
    model = model.cuda()
  
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad = True)
    
    best_metric = 0
    for epoch in range(1,epochs+1):
        print(epoch)
        train( model, device, train_generator, optimizer, epoch, writer)
        metric = test(model, device, val_generator, epoch, writer)
        if(metric>best_metric):
            best_metric = metric
            torch.save(model, "results/saved_models/transfer/"+name+".pth")
            print(f"Saved best model (metric={best_metric:.4f})")
            



if __name__ == '__main__':
    main()