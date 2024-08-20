### ABOUT THIS PROGRAM: Helps display each of the DINOv2 features for each image in a more appealing way

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

### HELPER CODE AND FUNCTIONS:

# Valid poses: ["male_thin", "male_wide", "female_thin", "female_wide"]

def display(image_number: int, target_pose: str, layer_number: int): # only top and upper poses available

    # First creates the figure
    fig = plt.figure(figsize=(30, 21)) # maybe enlarge

    # 2. Sets the values of rows and column variables 
    rows = 5
    columns = 5

    # 3. Reading images 

    # Originals
    selfie_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/selfie_faces/image" + str(image_number) + "_selfie_face.png").convert('RGB')
    unihuman_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/unihuman_faces/image" + str(image_number) + "_" + str(target_pose) + "_unihuman_face.png").convert('RGB')
    swapped_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/swapped_faces/image" + str(image_number) + "_" + str(target_pose) + "_swapped_face.png").convert('RGB')
    target_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/target_faces/" + str(target_pose) + "_face.png").convert('RGB') 
    texture_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/texture_map_faces/image" + str(image_number) + "_" + str(target_pose) + "_texture_map_face.png").convert('RGB')

    # UMAP 100 Eigenvectors
    umap_100_selfie_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_selfie_faces/Selfie_Face" + str(image_number) + "_Layer" + str(layer_number) + "_100Eigenvectors_UMAP_DINOv2.jpg").convert('RGB') 
    umap_100_unihuman_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_unihuman_faces/UniHuman_Face" + str(image_number) + "_" + str(target_pose) +  "_Layer" + str(layer_number) + "_100Eigenvectors_UMAP_DINOv2.jpg").convert('RGB') 
    umap_100_swapped_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_swapped_faces/Swapped_Face" + str(image_number) + "_" + str(target_pose) +  "_Layer" + str(layer_number) + "_100Eigenvectors_UMAP_DINOv2.jpg").convert('RGB') 
    umap_100_target_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_target_faces/" + str(target_pose) + "_top_Layer" + str(layer_number) + "_100Eigenvectors_UMAP_DINOv2.jpg").convert('RGB')
    umap_100_texture_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_texture_faces/Texture_Face" + str(image_number) + "_" + str(target_pose) + "_Layer" + str(layer_number) + "_100Eigenvectors_UMAP_DINOv2.jpg").convert('RGB')

    # UMAP 200 Eigenvectors
    umap_200_selfie_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_selfie_faces/Selfie_Face" + str(image_number) + "_Layer" + str(layer_number) + "_200Eigenvectors_UMAP_DINOv2.jpg").convert('RGB') 
    umap_200_unihuman_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_unihuman_faces/UniHuman_Face" + str(image_number) + "_" + str(target_pose) +  "_Layer" + str(layer_number) + "_200Eigenvectors_UMAP_DINOv2.jpg").convert('RGB') 
    umap_200_swapped_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_swapped_faces/Swapped_Face" + str(image_number) + "_" + str(target_pose) +  "_Layer" + str(layer_number) + "_200Eigenvectors_UMAP_DINOv2.jpg").convert('RGB') 
    umap_200_target_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_target_faces/" + str(target_pose) + "_top_Layer" + str(layer_number) + "_200Eigenvectors_UMAP_DINOv2.jpg").convert('RGB')
    umap_200_texture_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_texture_faces/Texture_Face" + str(image_number) + "_" + str(target_pose) + "_Layer" + str(layer_number) + "_200Eigenvectors_UMAP_DINOv2.jpg").convert('RGB')


    # tSNE 100 Eigenvectors
    tSNE_100_selfie_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_selfie_faces/Selfie_Face" + str(image_number) + "_Layer" + str(layer_number) + "_100Eigenvectors_tSNE_DINOv2.jpg").convert('RGB') 
    tSNE_100_unihuman_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_unihuman_faces/UniHuman_Face" + str(image_number) + "_" + str(target_pose) +  "_Layer" + str(layer_number) + "_100Eigenvectors_tSNE_DINOv2.jpg").convert('RGB') 
    tSNE_100_swapped_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_swapped_faces/Swapped_Face" + str(image_number) + "_" + str(target_pose) +  "_Layer" + str(layer_number) + "_100Eigenvectors_tSNE_DINOv2.jpg").convert('RGB') 
    tSNE_100_target_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_target_faces/" + str(target_pose) + "_top_Layer" + str(layer_number) + "_100Eigenvectors_tSNE_DINOv2.jpg").convert('RGB')
    tSNE_100_texture_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_texture_faces/Texture_Face" + str(image_number) + "_" + str(target_pose) + "_Layer" + str(layer_number) + "_100Eigenvectors_tSNE_DINOv2.jpg").convert('RGB')


    # tSNE 200 Eigenvectors
    tSNE_200_selfie_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_selfie_faces/Selfie_Face" + str(image_number) + "_Layer" + str(layer_number) + "_200Eigenvectors_tSNE_DINOv2.jpg").convert('RGB') 
    tSNE_200_unihuman_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_unihuman_faces/UniHuman_Face" + str(image_number) + "_" + str(target_pose) +  "_Layer" + str(layer_number) + "_200Eigenvectors_tSNE_DINOv2.jpg").convert('RGB') 
    tSNE_200_swapped_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_swapped_faces/Swapped_Face" + str(image_number) + "_" + str(target_pose) +  "_Layer" + str(layer_number) + "_200Eigenvectors_tSNE_DINOv2.jpg").convert('RGB') 
    tSNE_200_target_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_target_faces/" + str(target_pose) + "_top_Layer" + str(layer_number) + "_200Eigenvectors_tSNE_DINOv2.jpg").convert('RGB')
    tSNE_200_texture_face = Image.open("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_texture_faces/Texture_Face" + str(image_number) + "_" + str(target_pose) + "_Layer" + str(layer_number) + "_200Eigenvectors_tSNE_DINOv2.jpg").convert('RGB')

    # 4. Adds original subplots at corresponding positions -----
    fig.add_subplot(rows, columns, 1) 
    plt.imshow(selfie_face) # Showing selfie_face image 
    plt.axis('off') 
    plt.title("original selfie_face")

    fig.add_subplot(rows, columns, 2) 
    plt.imshow(unihuman_face) # Showing unihuman_face image 
    plt.axis('off') 
    plt.title("original unihuman_face")

    fig.add_subplot(rows, columns, 3) 
    plt.imshow(swapped_face) # Showing swapped_face image 
    plt.axis('off') 
    plt.title("original swapped_face")

    fig.add_subplot(rows, columns, 4) 
    plt.imshow(target_face) # Showing target_face image 
    plt.axis('off') 
    plt.title("original target_face")

    fig.add_subplot(rows, columns, 5) 
    plt.imshow(texture_face) # Showing texture_face image 
    plt.axis('off') 
    plt.title("original texture_face")

    # 5. Adds the UMAP 100 Eigenvector subplots at corresponding positions -----
    fig.add_subplot(rows, columns, 6) 
    plt.imshow(umap_100_selfie_face) # Showing selfie_face image 
    plt.axis('off') 
    plt.title("umap 100 selfie_face")

    fig.add_subplot(rows, columns, 7) 
    plt.imshow(umap_100_unihuman_face) # Showing unihuman_face image 
    plt.axis('off') 
    plt.title("umap 100 unihuman_face")

    fig.add_subplot(rows, columns, 8) 
    plt.imshow(umap_100_swapped_face) # Showing swapped_face image 
    plt.axis('off') 
    plt.title("umap 100 swapped_face")

    fig.add_subplot(rows, columns, 9) 
    plt.imshow(umap_100_target_face) # Showing target_face image 
    plt.axis('off') 
    plt.title("umap 100 target_face")

    fig.add_subplot(rows, columns, 10) 
    plt.imshow(umap_100_texture_face) # Showing texture_face image 
    plt.axis('off') 
    plt.title("umap 100 texture_face")

    # 6. Adds the UMAP 200 Eigenvector subplots at corresponding positions -----
    fig.add_subplot(rows, columns, 11) 
    plt.imshow(umap_200_selfie_face) # Showing selfie_face image 
    plt.axis('off') 
    plt.title("umap 200 selfie_face")

    fig.add_subplot(rows, columns, 12) 
    plt.imshow(umap_200_unihuman_face) # Showing unihuman_face image 
    plt.axis('off') 
    plt.title("umap 200 unihuman_face")

    fig.add_subplot(rows, columns, 13) 
    plt.imshow(umap_200_swapped_face) # Showing swapped_face image 
    plt.axis('off') 
    plt.title("umap 200 swapped_face")

    fig.add_subplot(rows, columns, 14) 
    plt.imshow(umap_200_target_face) # Showing target_face image 
    plt.axis('off') 
    plt.title("umap 200 target_face")

    fig.add_subplot(rows, columns, 15) 
    plt.imshow(umap_200_texture_face) # Showing texture_face image 
    plt.axis('off') 
    plt.title("umap 200 texture_face")

    # 7. Adds the tSNE 100 Eigenvector subplots at corresponding positions -----
    fig.add_subplot(rows, columns, 16) 
    plt.imshow(tSNE_100_selfie_face) # Showing selfie_face image 
    plt.axis('off') 
    plt.title("tSNE 100 selfie_face")

    fig.add_subplot(rows, columns, 17) 
    plt.imshow(tSNE_100_unihuman_face) # Showing unihuman_face image 
    plt.axis('off') 
    plt.title("tSNE 100 unihuman_face")

    fig.add_subplot(rows, columns, 18) 
    plt.imshow(tSNE_100_swapped_face) # Showing swapped_face image 
    plt.axis('off') 
    plt.title("tSNE 100 swapped_face")

    fig.add_subplot(rows, columns, 19) 
    plt.imshow(tSNE_100_target_face) # Showing target_face image 
    plt.axis('off') 
    plt.title("tSNE 100 target_face")

    fig.add_subplot(rows, columns, 20) 
    plt.imshow(tSNE_100_texture_face) # Showing texture_face image 
    plt.axis('off') 
    plt.title("tSNE 100 texture_face")

    # 6. Adds the tSNE 200 Eigenvector subplots at corresponding positions -----
    fig.add_subplot(rows, columns, 21) 
    plt.imshow(tSNE_200_selfie_face) # Showing selfie_face image 
    plt.axis('off') 
    plt.title("tSNE 200 selfie_face")

    fig.add_subplot(rows, columns, 22) 
    plt.imshow(tSNE_200_unihuman_face) # Showing unihuman_face image 
    plt.axis('off') 
    plt.title("tSNE 200 unihuman_face")

    fig.add_subplot(rows, columns, 23) 
    plt.imshow(tSNE_200_swapped_face) # Showing swapped_face image 
    plt.axis('off') 
    plt.title("tSNE 200 swapped_face")

    fig.add_subplot(rows, columns, 24) 
    plt.imshow(tSNE_200_target_face) # Showing target_face image 
    plt.axis('off') 
    plt.title("tSNE 200 target_face")

    fig.add_subplot(rows, columns, 25) 
    plt.imshow(tSNE_200_texture_face) # Showing texture_face image 
    plt.axis('off') 
    plt.title("tSNE 200 texture_face")

    plt.suptitle("Image" + str(image_number) + "_" + str(target_pose) + "_Layer" + str(layer_number))

    # 10. Saves the Matplot figure!
    plt.savefig("/home/kate/Selfie2Portrait/ncut_pytorch/compilations/Image" + str(image_number) + "_" + str(target_pose) + "_Layer" + str(layer_number) + "_Compilation.jpg")

### AUTOMATED CODE

images = [14, 23, 39]
poses = ["male_thin", "male_wide", "female_thin", "female_wide"]

for image_number in images:
    for pose in poses:
        for layer_number in range(0, 12):
            display(image_number, pose, layer_number)





