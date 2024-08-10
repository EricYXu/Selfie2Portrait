# ABOUT THE PROGRAM: Creates different types of masks for a selfie image, resizes the original image, and saves them to files!
import torch
import pickle
import os
import sys
import cv2
import numpy as np
from numpy import asarray
from PIL import Image
from math import *

### HELPER FUNCTIONS FOR MASK SEGMENTATION!!!

# Places the cropped arm mask 2D array into the proper 2D array that matches that of the original image
def overlap(input_array, top_left_x, top_left_y, bottom_right_x, bottom_right_y, original_width, original_height): # exterior for loop = height, interior for loop = width
    overlapped_array = np.zeros((original_height, original_width, 3), dtype=np.uint8) # all the added array entries are automatically set to black (convenient!)
    
    # Fills in the input_array into the new overlapping array
    for i in range(0, (bottom_right_y - top_left_y) - 1): # used to be (bottom_right_y - top_left_y) - 1
        for j in range(0, (bottom_right_x - top_left_x) - 1):
            overlapped_array[i + top_left_y][j + top_left_x] = input_array[i][j] # Fix this to transfer more accurately
    return overlapped_array

# Places the color-inverted cropped arm mask 2D array into the proper 2D array that matches that of the original image
def background_overlap(input_array, top_left_x, top_left_y, bottom_right_x, bottom_right_y, original_width, original_height): # exterior for loop = height, interior for loop = width
    # background_overlapped_array = np.zeros((original_height, original_width, 3), dtype=np.uint8) 
    background_overlapped_array = np.full((original_height, original_width, 3), 255, dtype=np.uint8) # all the added array entries are automatically set to white (convenient!)

    # Fills in the input_array into the new overlapping array
    for i in range(0, (bottom_right_y - top_left_y) - 1): # used to be (bottom_right_y - top_left_y) - 1
        for j in range(0, (bottom_right_x - top_left_x) - 1):
            background_overlapped_array[i + top_left_y][j + top_left_x] = input_array[i][j] # Fix this to transfer more accurately
    return background_overlapped_array

# Adding inpainting regions above and below the main image
def resize(input_array: list, input_width:int, input_height:int, square_dimension: int) -> list:
    resized_image_array = np.zeros((square_dimension, square_dimension, 3), dtype=np.uint8)

    for i in range(0, int((square_dimension - input_height)/2)):
        for j in range(0, input_width):
            resized_image_array[i][j] = (255,255,255) # sets the extra space to white color

    # Insert input from non-square image
    for i in range(int((square_dimension - input_height)/2), square_dimension - int((square_dimension - input_height)/2) - 1): # NOTE: Should I get rid of the -1? * * *
        for j in range(0, input_width):
            resized_image_array[i][j] = input_array[i - int((square_dimension - input_height)/2)][j]

    for i in range(int(square_dimension/2) + int(input_height/2), square_dimension):
        for j in range(0, input_width):
            resized_image_array[i][j] = (255,255,255) # sets the extra space to white color
    
    return resized_image_array

# Creates and saves the side rectangle mask for a given image array
def get_side_rectangle_mask(input_array, left_bound, right_bound):
    side_rectangle_mask = input_array
    for i in range(0, len(side_rectangle_mask)): # height
        for j in range(len(side_rectangle_mask[0])): # width
            if(j <= left_bound):
                side_rectangle_mask[i][j] = white_array
            elif(j >= right_bound):
                side_rectangle_mask[i][j] = white_array

    return side_rectangle_mask

# Creates and saves the blocky mask for a given image array
def get_blocky_mask(input_array, left_bound, right_bound, upper_bound, lower_bound): 
    blocky_mask = input_array
    for i in range(0, len(blocky_mask)): # height
        for j in range(len(blocky_mask[0])): # width
            if(i >= upper_bound):
                if(j <= left_bound):
                    blocky_mask[i][j] = white_array
                elif(j >= right_bound):
                    blocky_mask[i][j] = white_array

    return blocky_mask


### AUTOMATED PROCESS FOR ALL 154 IMAGES!!!
for image_number in range(1,155):
    os.system("python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \/mnt/volume2/Data/UnselfieData/selfies/image"+ str(image_number) + ".jpg --output results.pt -v")

    # Loading the data from the results.pkl file -> produces list of dictionaries for each image
    sys.path.append("/home/kate/Unselfie/detectron2/projects/DensePose/")
    f = open('/home/kate/Unselfie/detectron2/projects/DensePose/results.pt', 'rb')
    data = torch.load(f)

    # Loading the original image 
    original_image = Image.open("/mnt/volume2/Data/UnselfieData/selfies/image" + str(image_number) + ".jpg")

    # Converting the original PIL Image to a NumPy array
    original_image_array = asarray(original_image)
    initial_width = original_image_array.shape[1]
    initial_height = original_image_array.shape[0]

    # Extracting the labels from the dictionary + getting the coordinates for the bounding box
    img_labels = data[0]["pred_densepose"][0].labels # why can't you use the a different index number (error for 2, okay for 0)?
    proper_dimension_tensor = data[0]["pred_boxes_XYXY"][0] # these are the bounding box COORDINATES!!!
    # First two terms indicate the x- and y- coordinates of the top-left corner of the bounding box; the last two terms indicate the x,y coordinates of the bottom-right corner of the bounding box

    # Obtain the coordinates for the top-left and bottom-right corners of the bounding box
    top_left_x_coord = floor(proper_dimension_tensor[0].detach().numpy())
    top_left_y_coord = floor(proper_dimension_tensor[1].detach().numpy())
    bottom_right_x_coord = floor(proper_dimension_tensor[2].detach().numpy())
    bottom_right_y_coord = floor(proper_dimension_tensor[3].detach().numpy())

    # Saves the initial body mask image
    labels_np = (10 * img_labels).cpu().numpy().astype(np.uint8) # (10 * img_labels before)
    # output_img = Image.fromarray(labels_np)
    # output_img.save("/home/kate/Unselfie/detectron2/projects/DensePose/output_img.png") # --> IMAGE**

    print("Now to arm segmentation!")

    # Arm Segmentation
    labels_rgb = np.zeros((labels_np.shape[0], labels_np.shape[1], 3), dtype=np.uint8)
    mask = (labels_np >= 150) & (labels_np <= 210) # does this basically determine whether a value in the initial array is within a range and then print as pure white/black
    handmask = (labels_np >= 30) & (labels_np <= 50) # >= 30, <= 50; this essentially converts the values inside the tensor into a true/false boolean based on how bright it is in the initial mask

    labels_rgb[mask] = (255, 255, 255) # fills in cells with proper values white
    labels_rgb[handmask] = (255, 255, 255) # fills in cells with proper values white

    # Saving the arm segmentation mask
    arm_labels_img = Image.fromarray(labels_rgb) # this is the arm segmentation mask
    # arm_labels_img.save("/home/kate/Unselfie/detectron2/projects/DensePose/arm_labels_img.png") --> IMAGE**

    # Dilating the mask
    dilated_arm_mask = labels_rgb.copy()
    kernel = np.ones((15, 15), np.uint8) # TINKER WITH THIS!!! maybe try (25,25) --> originally (5,5) --> maybe (15, 15) is better
    dilated_arm_mask = cv2.dilate(dilated_arm_mask, kernel, iterations=3).astype(np.uint8)

    print("Now to conforming the arm_mask to the original dimensions!")

    arm_overlapped_array = overlap(dilated_arm_mask, top_left_x_coord, top_left_y_coord, bottom_right_x_coord, bottom_right_y_coord, initial_width, initial_height)
    side_rectangle_overlapped_array = arm_overlapped_array.copy()
    blocky_overlapped_array = arm_overlapped_array.copy()

    # # Save the overlapped array image
    # temp_overlapped_image = Image.fromarray(side_rectangle_overlapped_array)
    # temp_overlapped_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/temp_overlapped_img.png") # --> IMAGE**

    print("Now to background segmentation!")

    ## BACKGROUND SEGMENTATION! * * *
    background_labels = np.zeros((labels_np.shape[0], labels_np.shape[1], 3), dtype=np.uint8)
    background_mask = (labels_np == 0) # this is the BACKGROUND MASK TO RE-INPAINT BACKGROUND!!! * * *
    background_labels[background_mask] = (255, 255, 255)

    background_array = background_labels.copy() # copies the background mask to array form
    background_overlapped_array = background_overlap(background_array, top_left_x_coord, top_left_y_coord, bottom_right_x_coord, bottom_right_y_coord, initial_width, initial_height)

    ## FACE SEGMENTATION! * * *
    face_labels = np.zeros((labels_np.shape[0], labels_np.shape[1], 3), dtype=np.uint8)
    facemask = (labels_np >= 230) & (labels_np <= 270)
    face_labels[facemask] = (255, 255, 255) 

    # Dilate the face mask a bit!
    dilated_face_mask = face_labels.copy()
    face_kernel = np.ones((20, 20), np.uint8) 
    dilated_face_mask = cv2.dilate(dilated_face_mask, face_kernel, iterations=3).astype(np.uint8)

    # Saves the FACE MASK
    face_overlapped_array = overlap(dilated_face_mask, top_left_x_coord, top_left_y_coord, bottom_right_x_coord, bottom_right_y_coord, initial_width, initial_height)

    ########## For the first three masks

    # Creating scaffold for new square image
    square_dim = max(initial_width, initial_height)
    new_image_array = np.zeros((square_dim, square_dim, 3), dtype=np.uint8)

    # Finding the bounds for the rectangle masks
    left_bound = 0
    right_bound = 0
    previous_white = False
    no_white = True
    white_array = np.array([255, 255, 255])

    ### FINDS THE HORIZONTAL BOUNDS FOR THE RECTANGULAR MASK!
    for i in range(0, initial_width): # finding left_bound to make left square mask --> should be [0 --> initial_width]
        for j in range(0, initial_height):
            if(np.array_equal(side_rectangle_overlapped_array[j][i], white_array)):
                no_white = False
        if(no_white == True and previous_white == True):
            left_bound = i    
            break
        if(no_white == False and previous_white == False):
            previous_white = True
        no_white = True
            
    no_white = True
    for i in range(left_bound, initial_width): # finds the first occurence of white after the left_bound
        for j in range(0, initial_height): # finding right_bound to make right square mask
            if(np.array_equal(side_rectangle_overlapped_array[j][i], white_array)):
                no_white = False
        if(no_white == False):
            right_bound = i
            break

    ### FINDS THE VERTICAL BOUNDS FOR THE BLOCKY MASK!
    upper_bound = 0
    lower_bound = 0
    no_white = True
    previous_white = False

    for i in range(0, initial_height):
        for j in range(0, initial_width):
            if(np.array_equal(blocky_overlapped_array[i][j], white_array)):
                no_white = False
        if(no_white == False):
            upper_bound = i
            break

    no_white = True
    previous_white = False

    for i in range(upper_bound, initial_height): # finding left_bound to make left square mask --> should be [0 --> initial_width]
        for j in range(0, initial_width):
            if(np.array_equal(blocky_overlapped_array[i][j], white_array)):
                no_white = False
        if(no_white == True and previous_white == True):
            lower_bound = i
            break
        if(no_white == False and previous_white == False):
            previous_white = True
        no_white = True

    # MASK 1: TIGHT MASK----------

    # This saves the arm segmentation mask resized to a square and saved as a PIL Image
    arm_mask = resize(arm_overlapped_array, initial_width, initial_height, square_dim)
    final_arm_mask_image = Image.fromarray(arm_mask)
    final_arm_mask_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/Tight_Mask_Results/Tight_Mask_Image" + str(image_number) + ".png")

    # MASK 2: SIDE RECTANGLE MASK----------

    side_rectangle_mask = get_side_rectangle_mask(side_rectangle_overlapped_array, left_bound, right_bound)
    side_rectangle_mask = resize(side_rectangle_mask, initial_width, initial_height, square_dim)
    final_side_rectangle_mask_image = Image.fromarray(side_rectangle_mask)
    final_side_rectangle_mask_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/Side_Rect_Mask_Results/Side_Rect_Mask_Image" + str(image_number) + ".png")

    # MASK 3: BLOCKY MASK---------- 

    blocky_mask = get_blocky_mask(blocky_overlapped_array, left_bound, right_bound, upper_bound, lower_bound)
    blocky_mask = resize(blocky_mask, initial_width, initial_height, square_dim)
    final_blocky_mask_image = Image.fromarray(blocky_mask)
    final_blocky_mask_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/Blocky_Mask_Results/Blocky_Mask_Image" + str(image_number) + ".png") 

    # MASK 4: BACKGROUND MASK----------
    final_background_mask = resize(background_overlapped_array, initial_width, initial_height, square_dim)
    final_background_mask_image = Image.fromarray(final_background_mask)
    final_background_mask_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/Background_Mask_Results/Background_Mask_Image" + str(image_number) + ".png")

    # MASK 5: FACE MASK----------
    resized_face_array = resize(face_overlapped_array, initial_width, initial_height, square_dim)
    final_face_mask_image = Image.fromarray(resized_face_array)
    final_face_mask_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/Face_Masks/Face_Mask_Image" + str(image_number) + ".png")


    # Resize the image to a square in a similar fashion to the arm segmentation mask --> NOT A MASK
    bigger_square_dimension = max(original_image_array.shape[0], original_image_array.shape[1])
    resized_original_image_array = resize(original_image_array, initial_width, initial_height, bigger_square_dimension)
    final_resized_original_image = Image.fromarray(resized_original_image_array)
    final_resized_original_image.save("/home/kate/Unselfie/detectron2/projects/DensePose/Resized_Images/Resized_Image" + str(image_number) + ".png")

    print("Obtained masks for image" + str(image_number) + ".png")

print("Automated process complete!")