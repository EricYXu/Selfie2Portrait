### ABOUT THE PROGRAM: Obtains the pose background masks for inpainting after pose transfer!

import pickle
import torch
import os
import sys
import cv2
import numpy as np
from numpy import asarray
from PIL import Image
from math import *

## TODO: 1. Use an array of strings to automate this a bit better when additional target poses are included!

### HELPER CODE AND FUNCTIONS

# Places the cropped arm mask 2D array into the proper 2D array that matches that of the original image
def overlap(input_array, top_left_x, top_left_y, bottom_right_x, bottom_right_y, original_width, original_height): # exterior for loop = height, interior for loop = width
    overlapped_array = np.zeros((original_height, original_width, 3), dtype=np.uint8) # all the added array entries are automatically set to black (convenient!)
    
    # Fills in the input_array into the new overlapping array
    for i in range(0, (bottom_right_y - top_left_y) - 1): # used to be (bottom_right_y - top_left_y) - 1
        for j in range(0, (bottom_right_x - top_left_x) - 1):
            overlapped_array[i + top_left_y][j + top_left_x] = input_array[i][j] # Fix this to transfer more accurately
    return overlapped_array

# Adding inpainting regions above and below the main image
def resize(input_array: list, input_width:int, input_height:int, square_dimension: int) -> list:
    resized_image_array = np.zeros((square_dimension, square_dimension, 3), dtype=np.uint8)

    for i in range(0, int((square_dimension - input_height)/2)):
        for j in range(0, input_width):
            resized_image_array[i][j] = (255,255,255) # sets the extra space to white color

    # Insert input from non-square image
    for i in range(int((square_dimension - input_height)/2), square_dimension - int((square_dimension - input_height)/2)): # NOTE: THERE USED TO BE A "-1" after the second term in range()
        for j in range(0, input_width):
            resized_image_array[i][j] = input_array[i - int((square_dimension - input_height)/2)][j]

    for i in range(int(square_dimension/2) + int(input_height/2), square_dimension):
        for j in range(0, input_width):
            resized_image_array[i][j] = (255,255,255) # sets the extra space to white color
    
    return resized_image_array

# Inverts the black and white target pose to a background mask
def invert(input_array: list) -> list:
    inverted_array = input_array.copy()
    black_array = np.array([0, 0, 0])

    for height in range(0, inverted_array.shape[0]):
        for width in range(0, inverted_array.shape[1]):
            if(np.array_equal(inverted_array[height][width], black_array)):
                inverted_array[height][width] = (255, 255, 255)
            else:
                inverted_array[height][width] = (0, 0, 0)

    return inverted_array

### AUTOMATED PROCESS CODE

# TASK 1: MALE_THIN TARGET IMAGE----------

os.system("python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \/home/kate/Unselfie/UniHuman/code/Target_Images/male_thin_horizontal_target.jpg --output male_thin_results.pt -v")

# Loading the data from the results.pkl file -> produces list of dictionaries for each image
sys.path.append("/home/kate/Unselfie/detectron2/projects/DensePose/")
f = open('/home/kate/Unselfie/detectron2/projects/DensePose/male_thin_results.pt', 'rb')
data = torch.load(f)

# Obtaining data about male_thin image
original_male_thin_target = Image.open("/home/kate/Unselfie/UniHuman/code/Target_Images/male_thin_horizontal_target.jpg")
original_male_thin_array = asarray(original_male_thin_target)
male_thin_initial_width = original_male_thin_array.shape[1]
male_thin_initial_height = original_male_thin_array.shape[0]

# Obtaining labels from dictionary form
male_thin_img_labels = data[0]["pred_densepose"][0].labels
proper_dimension_tensor = data[0]["pred_boxes_XYXY"][0]

# Getting the coordinates of the bounding box
top_left_x_coord = floor(proper_dimension_tensor[0].detach().numpy())
top_left_y_coord = floor(proper_dimension_tensor[1].detach().numpy())
bottom_right_x_coord = floor(proper_dimension_tensor[2].detach().numpy())
bottom_right_y_coord = floor(proper_dimension_tensor[3].detach().numpy())

# Manipulating img_labels
male_thin_labels_np = (10 * male_thin_img_labels).cpu().numpy().astype(np.uint8)

os.system("clear")
print("Now onto full-body segmentation!")

# Getting the full-body segmentation!
male_thin_img_labels_rgb = np.zeros((male_thin_labels_np.shape[0], male_thin_labels_np.shape[1], 3), dtype=np.uint8)
male_thin_body_mask = (male_thin_labels_np > 0)
male_thin_img_labels_rgb[male_thin_body_mask] = (255, 255, 255)

# Dilates the male_thin body mask
male_thin_body_img = Image.fromarray(male_thin_img_labels_rgb)
dilated_male_thin_body_mask = male_thin_img_labels_rgb.copy()
kernel = np.ones((25, 25), np.uint8)
cv2.dilate(dilated_male_thin_body_mask, kernel, iterations=3).astype(np.uint8)

print("Now changing the mask dimensions and inverting color!")

male_thin_overlapped_array = overlap(dilated_male_thin_body_mask, top_left_x_coord, top_left_y_coord, bottom_right_x_coord, bottom_right_y_coord, male_thin_initial_width, male_thin_initial_height)
male_thin_background_array = invert(male_thin_overlapped_array)

# Resizing the image to 612 x 408 pixels
male_thin_temp_image = Image.fromarray(male_thin_background_array)
male_thin_temp_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/male_thin_temp_image.jpg") # good here
old_male_thin_temp_image = Image.open("/home/kate/Unselfie/detectron2/projects/DensePose/male_thin_temp_image.jpg").resize((612, 408))

# Resizing to 612 x 612 pixels
male_thin_temp_array = asarray(old_male_thin_temp_image)
resized_male_thin_background_array = resize(male_thin_temp_array, 612, 408, 612)

# Save the result as a PIL Image
final_male_thin_background_mask = Image.fromarray(resized_male_thin_background_array)
final_male_thin_background_mask.save("/home/kate/Unselfie/UniHuman/code/Background_Masks/male_thin_Background_Mask.jpg")

# TASK 2: MALE_WIDE TARGET IMAGE----------

os.system("python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \/home/kate/Unselfie/UniHuman/code/Target_Images/male_wide_horizontal_target.jpg --output male_wide_results.pt -v")

# Loading the data from the results.pkl file -> produces list of dictionaries for each image
sys.path.append("/home/kate/Unselfie/detectron2/projects/DensePose/")
f = open('/home/kate/Unselfie/detectron2/projects/DensePose/male_wide_results.pt', 'rb')
data = torch.load(f)

# Obtaining data about male_thin image
original_male_wide_target = Image.open("/home/kate/Unselfie/UniHuman/code/Target_Images/male_wide_horizontal_target.jpg")
original_male_wide_array = asarray(original_male_wide_target)
male_wide_initial_width = original_male_wide_array.shape[1]
male_wide_initial_height = original_male_wide_array.shape[0]

# Obtaining labels from dictionary form
male_wide_img_labels = data[0]["pred_densepose"][0].labels
proper_dimension_tensor = data[0]["pred_boxes_XYXY"][0]

# Getting the coordinates of the bounding box
top_left_x_coord = floor(proper_dimension_tensor[0].detach().numpy())
top_left_y_coord = floor(proper_dimension_tensor[1].detach().numpy())
bottom_right_x_coord = floor(proper_dimension_tensor[2].detach().numpy())
bottom_right_y_coord = floor(proper_dimension_tensor[3].detach().numpy())

# Manipulating img_labels
male_wide_labels_np = (10 * male_wide_img_labels).cpu().numpy().astype(np.uint8)

os.system("clear")
print("Now onto full-body segmentation!")

# Getting the full-body segmentation!
male_wide_img_labels_rgb = np.zeros((male_wide_labels_np.shape[0], male_wide_labels_np.shape[1], 3), dtype=np.uint8)
male_wide_body_mask = (male_wide_labels_np > 0)
male_wide_img_labels_rgb[male_wide_body_mask] = (255, 255, 255)

# Dilates the male_thin body mask
male_wide_body_img = Image.fromarray(male_wide_img_labels_rgb)
dilated_male_wide_body_mask = male_wide_img_labels_rgb.copy()
kernel = np.ones((25, 25), np.uint8)
cv2.dilate(dilated_male_wide_body_mask, kernel, iterations=3).astype(np.uint8)

print("Now changing the mask dimensions and inverting color!")

male_wide_overlapped_array = overlap(dilated_male_wide_body_mask, top_left_x_coord, top_left_y_coord, bottom_right_x_coord, bottom_right_y_coord, male_wide_initial_width, male_wide_initial_height)
male_wide_background_array = invert(male_wide_overlapped_array)

# Resizing the image to 612 x 408 pixels
male_wide_temp_image = Image.fromarray(male_wide_background_array)
male_wide_temp_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/male_wide_temp_image.jpg") # good here
old_male_wide_temp_image = Image.open("/home/kate/Unselfie/detectron2/projects/DensePose/male_wide_temp_image.jpg").resize((612, 408)) # used to be .resize((612, 408))

# Resizing to 612 x 612 pixels
male_wide_temp_array = asarray(old_male_wide_temp_image)
resized_male_wide_background_array = resize(male_wide_temp_array, 612, 408, 612)

# Save the result as a PIL Image
final_male_wide_background_mask = Image.fromarray(resized_male_wide_background_array)
final_male_wide_background_mask.save("/home/kate/Unselfie/UniHuman/code/Background_Masks/male_wide_Background_Mask.jpg")


# TASK 3: FEMALE_THIN TARGET IMAGE----------

os.system("python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \/home/kate/Unselfie/UniHuman/code/Target_Images/female_thin_horizontal_target.jpg --output female_thin_results.pt -v")

# Loading the data from the results.pkl file -> produces list of dictionaries for each image
sys.path.append("/home/kate/Unselfie/detectron2/projects/DensePose/")
f = open('/home/kate/Unselfie/detectron2/projects/DensePose/female_thin_results.pt', 'rb')
data = torch.load(f)

# Obtaining data about male_thin image
original_female_thin_target = Image.open("/home/kate/Unselfie/UniHuman/code/Target_Images/female_thin_horizontal_target.jpg")
original_female_thin_array = asarray(original_female_thin_target)
female_thin_initial_width = original_female_thin_array.shape[1]
female_thin_initial_height = original_female_thin_array.shape[0]

# Obtaining labels from dictionary form
female_thin_img_labels = data[0]["pred_densepose"][0].labels
proper_dimension_tensor = data[0]["pred_boxes_XYXY"][0]

# Getting the coordinates of the bounding box
top_left_x_coord = floor(proper_dimension_tensor[0].detach().numpy())
top_left_y_coord = floor(proper_dimension_tensor[1].detach().numpy())
bottom_right_x_coord = floor(proper_dimension_tensor[2].detach().numpy())
bottom_right_y_coord = floor(proper_dimension_tensor[3].detach().numpy())

# Manipulating img_labels
female_thin_labels_np = (10 * female_thin_img_labels).cpu().numpy().astype(np.uint8)

os.system("clear")
print("Now onto full-body segmentation!")

# Getting the full-body segmentation!
female_thin_img_labels_rgb = np.zeros((female_thin_labels_np.shape[0], female_thin_labels_np.shape[1], 3), dtype=np.uint8)
female_thin_body_mask = (female_thin_labels_np > 0)
female_thin_img_labels_rgb[female_thin_body_mask] = (255, 255, 255)

# Dilates the male_thin body mask
female_thin_body_img = Image.fromarray(female_thin_img_labels_rgb)
dilated_female_thin_body_mask = female_thin_img_labels_rgb.copy()
kernel = np.ones((25, 25), np.uint8)
cv2.dilate(dilated_female_thin_body_mask, kernel, iterations=3).astype(np.uint8)

print("Now changing the mask dimensions and inverting color!")

female_thin_overlapped_array = overlap(dilated_female_thin_body_mask, top_left_x_coord, top_left_y_coord, bottom_right_x_coord, bottom_right_y_coord, female_thin_initial_width, female_thin_initial_height)
female_thin_background_array = invert(female_thin_overlapped_array)

# Resizing the image to 612 x 408 pixels
female_thin_temp_image = Image.fromarray(female_thin_background_array)
female_thin_temp_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/female_thin_temp_image.jpg") # good here
old_female_thin_temp_image = Image.open("/home/kate/Unselfie/detectron2/projects/DensePose/female_thin_temp_image.jpg").resize((612, 408)) # used to be .resize((612, 408))

# Resizing to 612 x 612 pixels
female_thin_temp_array = asarray(old_female_thin_temp_image)
resized_female_thin_background_array = resize(female_thin_temp_array, 612, 408, 612)

# Save the result as a PIL Image
final_female_thin_background_mask = Image.fromarray(resized_female_thin_background_array)
final_female_thin_background_mask.save("/home/kate/Unselfie/UniHuman/code/Background_Masks/female_thin_Background_Mask.jpg")

# TASK 4: FEMALE_WIDE TARGET IMAGE----------

os.system("python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \/home/kate/Unselfie/UniHuman/code/Target_Images/female_wide_horizontal_target.jpg --output female_wide_results.pt -v")

# Loading the data from the results.pkl file -> produces list of dictionaries for each image
sys.path.append("/home/kate/Unselfie/detectron2/projects/DensePose/")
f = open('/home/kate/Unselfie/detectron2/projects/DensePose/female_wide_results.pt', 'rb')
data = torch.load(f)

# Obtaining data about male_thin image
original_female_wide_target = Image.open("/home/kate/Unselfie/UniHuman/code/Target_Images/female_wide_horizontal_target.jpg")
original_female_wide_array = asarray(original_female_wide_target)
female_wide_initial_width = original_female_wide_array.shape[1]
female_wide_initial_height = original_female_wide_array.shape[0]

# Obtaining labels from dictionary form
female_wide_img_labels = data[0]["pred_densepose"][0].labels
proper_dimension_tensor = data[0]["pred_boxes_XYXY"][0]

# Getting the coordinates of the bounding box
top_left_x_coord = floor(proper_dimension_tensor[0].detach().numpy())
top_left_y_coord = floor(proper_dimension_tensor[1].detach().numpy())
bottom_right_x_coord = floor(proper_dimension_tensor[2].detach().numpy())
bottom_right_y_coord = floor(proper_dimension_tensor[3].detach().numpy())

# Manipulating img_labels
female_wide_labels_np = (10 * female_wide_img_labels).cpu().numpy().astype(np.uint8)

os.system("clear")
print("Now onto full-body segmentation!")

# Getting the full-body segmentation!
female_wide_img_labels_rgb = np.zeros((female_wide_labels_np.shape[0], female_wide_labels_np.shape[1], 3), dtype=np.uint8)
female_wide_body_mask = (female_wide_labels_np > 0)
female_wide_img_labels_rgb[female_wide_body_mask] = (255, 255, 255)

# Dilates the male_thin body mask
female_wide_body_img = Image.fromarray(female_wide_img_labels_rgb)
dilated_female_wide_body_mask = female_wide_img_labels_rgb.copy()
kernel = np.ones((25, 25), np.uint8)
cv2.dilate(dilated_female_wide_body_mask, kernel, iterations=3).astype(np.uint8)

print("Now changing the mask dimensions and inverting color!")

female_wide_overlapped_array = overlap(dilated_female_wide_body_mask, top_left_x_coord, top_left_y_coord, bottom_right_x_coord, bottom_right_y_coord, female_wide_initial_width, female_wide_initial_height)
female_wide_background_array = invert(female_wide_overlapped_array)

# Resizing the image to 612 x 408 pixels
female_wide_temp_image = Image.fromarray(female_wide_background_array)
female_wide_temp_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/female_wide_temp_image.jpg") # good here
old_female_wide_temp_image = Image.open("/home/kate/Unselfie/detectron2/projects/DensePose/female_wide_temp_image.jpg").resize((612, 408)) # used to be resize((612, 408))

# Resizing to 612 x 612 pixels
female_wide_temp_array = asarray(old_female_wide_temp_image)
resized_female_wide_background_array = resize(female_wide_temp_array, 612, 408, 612)

# Save the result as a PIL Image
final_female_wide_background_mask = Image.fromarray(resized_female_wide_background_array)
final_female_wide_background_mask.save("/home/kate/Unselfie/UniHuman/code/Background_Masks/female_wide_Background_Mask.jpg")

print("Process complete!")
