### ABOUT THIS PROGRAM: Obtains the DINO v2 features of input selfies and UV-texture maps, then it displays it using PCA method

import os
import numpy as np
# from PIL import Image
import PIL.Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA # may need to install sklearn
# from dinov2.eval.setup import build_model_for_eval
# from dinov2.configs import load_and_merge_config

### HELPER CODE AND FUNCTIONS:

print("First message")

images = []
# patch_h = 40
# patch_w = 40
# feat_dim = 1536 # vitg14

### PREP

# 1. Load five input selfies and resize
for image_number in range(1, 5):
    input_selfie = cv2.imread("/mnt/volume2/Data/UnselfieData/selfies/image" + str(image_number) + ".jpg")
    input_selfie = cv2.resize(input_selfie, (448,448)) # (448, 448)
    input_selfie = cv2.cvtColor(input_selfie, cv2.COLOR_BGR2RGB)

    # 2. Transforming to fp32 
    input_selfie = input_selfie.astype('float32')/255
    images.append(input_selfie)
    plt.subplot(220+image_number) 

# 3. Placing inside input tensor
input_selfie_in_array = np.stack(images)
input_tensor = torch.Tensor(np.transpose(input_selfie_in_array, [0, 3, 2, 1]))

# 3. Image normalization for input tensor
transform = T.Compose([T.Normalize(mean=0.5, std=0.2)])
input_tensor = transform(input_tensor)

# 4. Load and run the DINOv2 giant model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14') # CURRENTLY vitg14; can try using vits14 or vitg14

print("Second Message")

result = model.forward_features(input_tensor)
patch_tokens = result['x_norm_patchtokens'].detach().numpy().reshape([4,1024,-1])

# 5. Visualize results using PCA -- TRAINING
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

fg_pca = PCA(n_components=1)

masks=[]
plt.figure(figsize=(10,10))

all_patches = patch_tokens.reshape([-1,1536]) # or [-1,1024]
reduced_patches = fg_pca.fit_transform(all_patches)
# Scale the feature to (0,1)
norm_patches = minmax_scale(reduced_patches)

# Reshape the feature value to the original image size
image_norm_patches = norm_patches.reshape([4,1024]) # [4,1024], [6, 1024] kinda works

for i in range(4):
    image_patches = image_norm_patches[i,:]

    # choose a threshold to segment the foreground
    mask = (image_patches > 0.6).ravel()
    masks.append(mask)

    image_patches[np.logical_not(mask)] = 0

    plt.subplot(221+i)
    plt.imshow(images[i])
    plt.imshow(image_patches.reshape([32,-1]).T, extent=(0,448,448,0), alpha=0.5)

### PCA VISUALIZATION TIME

object_pca = PCA(n_components=3)

# extract foreground patches
mask_indices = [0, *np.cumsum([np.sum(m) for m in masks]), -1]
fg_patches = np.vstack([patch_tokens[i,masks[i],:] for i in range(4)])

# fit PCA to foreground, scale each feature to (0,1)
reduced_patches = object_pca.fit_transform(fg_patches)
reduced_patches = minmax_scale(reduced_patches)

print(object_pca.explained_variance_ratio_)

# reshape the features to the original image size
plt.figure(figsize=(10,20))
for i in range(4):
    patch_image = np.zeros((1024,3), dtype='float32')
    patch_image[masks[i],:] = reduced_patches[mask_indices[i]:mask_indices[i+1],:]

    color_patches = patch_image.reshape([32,-1,3]).transpose([1,0,2])

    plt.subplot(421+(2*i))
    plt.imshow(color_patches)
    plt.savefig("/home/kate/Unselfie/dinov2/code/TEST_InputSelfie_DINO/PCA_ColorPatches_Image" + str(image_number) + "_InputSelfie_DINO.png")

    plt.subplot(421+(2*i)+1)
    plt.imshow(images[i])
    plt.savefig("/home/kate/Unselfie/dinov2/code/TEST_InputSelfie_DINO/PCA_otherimages_Image" + str(image_number) + "_InputSelfie_DINO.png")

for i in range(4):
    plt.subplot(425+i)
    plt.imshow

print("Training complete!")
print("Now testing with new input selfie!")

### TESTING DINOv2 FEATURE EXTRACTION AND PCA VISUALIZATION

test_number = 23

test_image = cv2.imread("/mnt/volume2/Data/UnselfieData/selfies/image" + str(test_number) + ".jpg")
test_image = cv2.resize(test_image, (672,672))  # unicorn: 224,280
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = test_image.astype('float32')/255

test_images = [test_image]

test_images_arr = np.stack(test_images)
test_tensor = torch.Tensor(np.transpose(test_images_arr, [0, 3, 2, 1]))
test_tensor = transform(test_tensor)

test_result = model.forward_features(test_tensor)

test_patch_tokens = test_result['x_norm_patchtokens'].detach().numpy().reshape([2304, -1])
fg_result = fg_pca.transform(test_patch_tokens)
fg_result = minmax_scale(fg_result)

fg_mask = (fg_result > 0) # TRY EDITING THIS, previously 0.5 --> TINKER HERE

# PCA Visualizing the Test Selfie

object_result = object_pca.transform(test_patch_tokens)
object_result = minmax_scale(object_result)

only_object = np.zeros_like(object_result)
only_object[fg_mask.ravel(), :] = object_result[fg_mask.ravel(), :]

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(test_image)
plt.subplot(122)
plt.imshow(only_object.reshape([48, -1, 3]).transpose([1, 0, 2]))
plt.savefig("/home/kate/Unselfie/dinov2/code/TEST_InputSelfie_DINO/PCA_TEST_Image" + str(test_number) + "_InputSelfie_DINO.png")

print("Overall process complete!")

exit()


### SCRAP


# # Sets up torch transform
# transform = T.Compose([
#     T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#  ])

# # Loads the input selfie and converts to tensor
# input_selfie = PIL.Image.open("/mnt/volume2/Data/UnselfieData/selfies/image" + str(image_number) + ".jpg")
# input_selfie_tensor = transform(input_selfie)[:3]

# ### FEATURE EXTRACTION

# # Extract features
# features = torch.zeros(4, patch_h * patch_w, feat_dim)
# imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14)

# with torch.no_grad():
#     features_dict = dinov2_vitg14.forward_features(imgs_tensor)
#     features = features_dict['x_norm_patchtokens']

# print("Feature extraction complete!")
# print("Now running PCA visualization script!")

# ### PCA Visualization

# features = features.reshape(4 * patch_h * patch_w, feat_dim)

# pca = PCA(n_components=3)
# pca.fit(features)
# pca_features = pca.transform(features)

# # visualize PCA components for finding a proper threshold
# plt.subplot(1, 3, 1)
# plt.hist(pca_features[:, 0])
# plt.savefig("/home/kate/Unselfie/dinov2/code/InputSelfie_DINO/PCA_Image" + str(image_number) + "_InputSelfie_DINO.png")

# plt.subplot(1, 3, 2)
# plt.hist(pca_features[:, 1])
# plt.subplot(1, 3, 3)
# plt.hist(pca_features[:, 2])
# plt.show()
# plt.close()

# # uncomment below to plot the first pca component
# pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(pca_features[i * patch_h * patch_w: (i+1) * patch_h * patch_w, 0].reshape(patch_h, patch_w))
# plt.show()
# plt.close()


# # # PCA for only foreground patches
# # pca.fit(features[pca_features_fg]) # NOTE: I forgot to add it in my original answer
# # pca_features_rem = pca.transform(features[pca_features_fg])
# # for i in range(3):
# #     # pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())
# #     # transform using mean and std, I personally found this transformation gives a better visualization
# #     pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].mean()) / (pca_features_rem[:, i].std() ** 2) + 0.5

# pca_features_rgb = pca_features.copy()
# pca_features_rgb[pca_features_bg] = 0
# pca_features_rgb[pca_features_fg] = pca_features_rem

# pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.imshow(pca_features_rgb[i][..., ::-1])
# plt.savefig("/home/kate/Unselfie/dinov2/code/InputSelfie_DINO/features_image" + str(image_number) + ".png")
# plt.show()
# plt.close()


# # # unsure of below
# # # pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
# # # pca_features = pca_features * 255

# # # plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
# # # plt.savefig("/home/kate/Unselfie/dinov2/code/InputSelfie_DINO/PCA_Image" + str(image_number) + "_InputSelfie_DINO.png")

# print("PCA Visualization complete!")


# ### SCRAP!

# # output_img = features[0]
# # output_img_array = (output_img).detach().cpu().numpy().astype(np.uint8)
# # output_display = PIL.Image.fromarray(output_img_array)
# # output_display.save("/home/kate/Unselfie/dinov2/code/InputSelfie_DINO/Image" + str(image_number) + "_InputSelfie_DINO.png")

