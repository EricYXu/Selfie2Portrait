### ABOUT THE PROGRAM: Tests the pose transfer process using the UniHuman model by extracting key information!

import os
import sys
import subprocess
import numpy as np
from PIL import Image
import cv2
import pickle

### BODY TEXTURE TESTING PORTION

pose_types = ["male_thin_top", "male_wide_top", "female_thin_top", "female_wide_top"]
# Removed upper poses for time efficiency after Image25 (Image26-Image154 do not have upper poses)

for image_number in range(128, 155):

    for pose_number in range(0, len(pose_types)):

        # First, open both the source_img_paths.txt and tgt_img_paths.txt files!
        source_file = open("source_img_paths.txt", "w")
        target_file = open("tgt_img_paths.txt", "w")

        # Adds entries to each txt file
        source_file.write("/home/kate/Selfie2Portrait/detectron2/projects/DensePose/Resized_Images/Resized_Image" + str(image_number) + ".png")
        target_file.write("/home/kate/Selfie2Portrait/UniHuman/code/Target_Images/" + str(pose_types[pose_number]) + "_target.jpg")

        # Close txt files to get UniHuman to work properly
        source_file.close()
        target_file.close()

        # Run Infer_Testing.py to extract NumPy body texture
        os.system("python Infer_Testing.py  --task reposing --src-img-list source_img_paths.txt --tgt-img-list tgt_img_paths.txt  --out-dir ./Body_Textures")

        # Extract the NumPy body texture array once the infer_testing script has been run
        pwarped_tex_img_np_extract = np.load("/home/kate/Selfie2Portrait/UniHuman/code/Body_Texture_Array.npy")

        # Display this extracted body texture as a PIL Image
        pwarped_tex_img_np_extract_image = Image.fromarray(pwarped_tex_img_np_extract)
        pwarped_tex_img_np_extract_image.save("/home/kate/Selfie2Portrait/UniHuman/code/Body_Textures/Image" + str(image_number) + "_" + str(pose_types[pose_number]) + "_body_texture.png")

        # Yay!
        print("Pose" + str(pose_number) + " complete!")

    print("PoseTransfer Testing for image" + str(image_number) + " complete!")







