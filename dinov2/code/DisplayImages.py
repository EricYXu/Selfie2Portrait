import sys
import numpy as np
import PIL
from PIL import Image

for image_number in range(1, 155):
    list_im = ["/mnt/volume2/Data/UnselfieData/selfies/image" + str(image_number) + ".jpg", "/home/kate/Selfie2Portrait/UniHuman/code/Body_Textures/Image" + str(image_number) + "_male_thin_top_body_texture.png", "/home/kate/Selfie2Portrait/dinov2/code/pca_results_for_input_selfies/image" + str(image_number) + ".jpg", "/home/kate/Selfie2Portrait/dinov2/code/pca_results_for_texture_maps/image" + str(image_number) + "_male_thin_top.jpg"]
    imgs    = [ Image.open(i) for i in list_im ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])

    # save that beautiful picture
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save("/home/kate/Selfie2Portrait/dinov2/code/compilations/Image" + str(image_number) + "_FeatureAndTexture_Compilation.jpg")  


# ###



# image_number = 5

# images = [Image.open(x) for x in ["/mnt/volume2/Data/UnselfieData/selfies/image" + str(image_number) + ".jpg", "/home/kate/Selfie2Portrait/UniHuman/code/Body_Textures/Image" + str(image_number) + "_male_thin_top_body_texture.png", "/home/kate/Selfie2Portrait/dinov2/code/pca_results_for_input_selfies/image" + str(image_number) + ".jpg", "/home/kate/Selfie2Portrait/dinov2/code/pca_results_for_texture_maps/image" + str(image_number) + "_male_thin_top.jpg"]]
# widths, heights = zip(*(i.size for i in images))

# total_width = sum(widths)
# max_height = max(heights)

# new_im = Image.new('RGB', (total_width, max_height))

# x_offset = 0
# for im in images:
#   new_im.paste(im, (x_offset,0))
#   x_offset += im.size[0]

# new_im.save("/home/kate/Selfie2Portrait/dinov2/code/compilations/Image" + str(image_number) + "_FeatureAndTexture_Compilation.jpg")
