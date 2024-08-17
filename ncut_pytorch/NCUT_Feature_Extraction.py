### ABOUT THIS PROGRAM: Uses the ncut_pytorch repo to extract more features from input selfies, UniHuman outputs, and texture maps

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from ncut_pytorch import quantile_normalize
from ncut_feature_extractors import image_dinov2_feature as feature_extractor
from ncut_pytorch import NCUT
from ncut_pytorch import quantile_normalize
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from ncut_pytorch import rgb_from_umap_sphere
from ncut_pytorch import rgb_from_tsne_3d

# Let's use process on images [5, 14, 23, 39, 63, 70, 72, 93]

image_number = 14
number_eigenvectors = 100

for layer_number in range(0, 12):
    UMAP = True
    tSNE = False

    # 1. Load the input image
    url = "/mnt/volume2/Data/UnselfieData/selfies/image" + str(image_number) + ".jpg"
    image = Image.open(url)

    # 2. Use DINOv2 features
    features = feature_extractor(image, resolution=(448, 448), layer=layer_number) # can change layer number here!!!
    h, w, c = features.shape

    # 3. Flatten the pixels into nodes
    n = h * w  # flatten the pixels into nodes
    features = features.reshape(n, c)
    model = NCUT(num_eig=number_eigenvectors) # INCREASE THIS NUMBER TO 100 -> 150 -> 200; TINKER HERE * * *
    eigenvectors, eigenvalues = model.fit_transform(features)

    # 4. 
    fig, axs = plt.subplots(3, 4, figsize=(13, 10))
    i_eig = 0
    for i_row in range(3):
        for i_col in range(1, 4):
            ax = axs[i_row, i_col]
            ax.imshow(eigenvectors[:, i_eig].reshape(h, w), cmap="coolwarm", vmin=-0.1, vmax=0.1)
            ax.set_title(f"lambda_{i_eig} = {eigenvalues[i_eig]:.3f}")
            ax.axis("off")
            i_eig += 1
    for i_row in range(3):
        ax = axs[i_row, 0]
        start, end = i_row * 3, (i_row + 1) * 3
        rgb = quantile_normalize(eigenvectors[:, start:end]).reshape(h, w, 3)
        ax.imshow(rgb)
        ax.set_title(f"eigenvectors {start}-{end-1}")
        ax.axis("off")
    plt.suptitle("Top 9 eigenvectors of Ncut DiNOv2 last layer features")
    plt.tight_layout()
    # plt.savefig("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features/TOP9Eigenvectors.jpg") # --> Likely don't need

    # 5. 
    def plot_3d_animation(X_3d, rgb, title):
        x, y, z = X_3d.T
        fig = plt.figure(figsize=(10, 5))

        # Add a subplot for the static image
        ax1 = fig.add_subplot(121)
        ax1.imshow(rgb.reshape(h, w, 3))
        ax1.axis('off')  # Hide axes

        # Add a subplot for the 3D scatter plot
        ax = fig.add_subplot(122, projection='3d')
        scat = ax.scatter(x, y, z, c=rgb, s=10)

        # set ticks labels
        ax.set_xlabel("Dimension #1")
        ax.set_ylabel("Dimension #2")
        ax.set_zlabel("Dimension #3")

        # set ticks, labels to none
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        z_ticks = ax.get_zticks()
        labels = ["" for _ in range(len(x_ticks))]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_zticklabels(labels)

        plt.suptitle(title)
        plt.show()

        # Define the update function for the animation
        def update(frame):
            if frame <= 360:
                ax.view_init(elev=10., azim=frame)
            if frame > 360 and frame <= 720:
                ax.view_init(elev=frame-360, azim=10.)

        # Create the animation
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 3), interval=60)
        from IPython.display import HTML
        html = HTML(ani.to_jshtml())
        display(html)

    # 6. 
    def plot_3d(X_3d, rgb, title):
        x, y, z = X_3d.T
        fig = plt.figure(figsize=(10, 5))

        # Add a subplot for the static image
        ax1 = fig.add_subplot(111)
        ax1.imshow(rgb.reshape(h, w, 3))
        ax1.axis('off')  # Hide axes

        # # Add a subplot for the 3D scatter plot --> NOT USING CURRENTLY
        # ax = fig.add_subplot(122, projection='3d')
        # scat = ax.scatter(x, y, z, c=rgb, s=10)

        # # set ticks labels
        # ax.set_xlabel("Dimension #1")
        # ax.set_ylabel("Dimension #2")
        # ax.set_zlabel("Dimension #3")

        # # set ticks, labels to none
        # x_ticks = ax.get_xticks()
        # y_ticks = ax.get_yticks()
        # z_ticks = ax.get_zticks()
        # labels = ["" for _ in range(len(x_ticks))]
        # ax.set_xticklabels(labels)
        # ax.set_yticklabels(labels)
        # ax.set_zticklabels(labels)

        # plt.suptitle(title)
        if(UMAP == True):
            plt.savefig("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_selfies/Selfie" + str(image_number) + "_Layer" + str(layer_number) + "_" + str(number_eigenvectors) + "Eigenvectors_UMAP_DINOv2.jpg")
            plt.savefig("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_selfies/Selfie" + str(image_number) + "_Layer" + str(layer_number) + "_" + str(number_eigenvectors) + "Eigenvectors_UMAP_DINOv2.jpg")
        elif(tSNE == True):
            plt.savefig("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_selfies/Selfie" + str(image_number) + "_Layer" + str(layer_number) + "_" + str(number_eigenvectors) + "Eigenvectors_tSNE_DINOv2.jpg")
            plt.savefig("/home/kate/Selfie2Portrait/ncut_pytorch/ncut_features_for_selfies/Selfie" + str(image_number) + "_Layer" + str(layer_number) + "_" + str(number_eigenvectors) + "Eigenvectors_tSNE_DINOv2.jpg")



    # 7. 
    X_3d, rgb = rgb_from_umap_sphere(eigenvectors[:, :10], device="cpu", n_neighbors=100, min_dist=0.1)
    plot_3d(X_3d, rgb, "Top " + str(number_eigenvectors) + " Ncut eigenvectors from DiNOv2 layer " + str(layer_number) + " features using UMAP")

    # 8.
    UMAP = False
    tSNE = True
    X_3d, rgb = rgb_from_tsne_3d(eigenvectors[:, :10], device="cpu", perplexity=100)
    plot_3d(X_3d, rgb, "Top " + str(number_eigenvectors) + " Ncut eigenvectors from DiNOv2 layer " + str(layer_number) + " features using tSNE")


print("Process complete!")



