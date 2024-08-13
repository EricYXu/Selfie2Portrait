### ABOUT THIS PROGRAM: Obtains the DINO v2 features of input selfies, then it displays it using PCA method --> ALTERNATE

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
import time

# Constructing the img_paths array
img_paths = []
for i in range(1, 155):
    img_paths.append("/mnt/volume2/Data/UnselfieData/selfies/image" + str(i) + ".jpg") # INPUT SELFIES
    

# Parameters
num_patches = 36
dino_model = "dinov2_vitb14"
feat_dim = 768

# Prepping the inputs
num_imgs = len(img_paths)
img_new_size = 14 * num_patches
transform = T.Compose([T.Resize(img_new_size), T.CenterCrop(img_new_size), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])

# Extract image features using DINOv2
dinov2 = torch.hub.load('facebookresearch/dinov2', dino_model)
features = torch.zeros(num_imgs, num_patches * num_patches, feat_dim)
imgs_tensor = torch.zeros(num_imgs, 3, num_patches * 14, num_patches * 14)

# Taking in input selfies
for i in range(num_imgs):
  img_path = img_paths[i]
  img = Image.open(img_path).convert('RGB')
  imgs_tensor[i] = transform(img)[:3]

with torch.no_grad():
  features_dict = dinov2.forward_features(imgs_tensor)
  features = features_dict['x_norm_patchtokens'] # TODO: 1. Look at other keys in the features_dict to see if other methods are better at extracting features

# Visualize features using PCA
features = features.reshape(num_imgs * num_patches * num_patches, feat_dim)
pca = PCA(n_components=3)
pca.fit(features)

pca_features = pca.transform(features)
pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
pca_features = pca_features * 255
pca_features = pca_features.reshape(num_imgs, num_patches, num_patches, 3)

# Saving the outputs
for i in range(num_imgs):
  plt.imshow(pca_features[i].astype(np.uint8))
  plt.savefig('/home/kate/Selfie2Portrait/dinov2/code/pca_results_for_input_selfies/image' + str(i + 1) + '.jpg')

# Displaying the Matplots in a more convenient way
## TODO

print("Process complete!")