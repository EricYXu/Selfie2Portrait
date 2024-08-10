### ABOUT THE PROGRAM: Automates the pose transfer process using the UniHuman model!

import os
import sys
import subprocess

### HELPER CODE AND FUNCTIONS:

pose_types = ["male_thin_top", "male_thin_upper", "male_thin_full", "male_wide_top", "male_wide_upper", "male_wide_full", "female_thin_top", "female_thin_upper", "female_thin_full", "female_wide_top", "female_wide_upper", "female_wide_full"]

### AUTOMATED PROCESS FOR POSE TRANSFER: 

for image_number in range(1, 155):

    for pose_number in range(0, len(pose_types)):

        # First, open both the source_img_paths.txt and tgt_img_paths.txt files!
        source_file = open("source_img_paths.txt", "w")
        target_file = open("tgt_img_paths.txt", "w")

        # Adds entries to each text file
        source_file.write("/home/kate/Unselfie/detectron2/projects/DensePose/Resized_Images/Resized_Image" + str(image_number) + ".png")
        target_file.write("/home/kate/Unselfie/UniHuman/code/Target_Images/" + str(pose_types[pose_number]) + "_target.jpg")

        source_file.close()
        target_file.close()

        os.system("python infer.py  --task reposing --src-img-list source_img_paths.txt --tgt-img-list tgt_img_paths.txt  --out-dir ./additional_resized_results")

        print("A pose transfer task complete! (" + str(pose_number + 1) + "/" + str(len(pose_types)) + ")")

    print("Process for resized_image" + str(image_number) + " complete!")

print("Automated process complete!")

