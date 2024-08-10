### ABOUT THIS PROGRAM: Uses the Open-CV python library to warp our target face onto the resized UniHuman result for better identity preservation!

import cv2
import PIL as Image
import numpy as np
import dlib

### HELPER CODE AND FUNCTIONS!!!

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

pose_types = ["male_thin_top", "male_thin_upper", "male_thin_full", "male_wide_top", "male_wide_upper", "male_wide_full", "female_thin_top", "female_thin_upper", "female_thin_full", "female_wide_top", "female_wide_upper", "female_wide_full"]

# ignore full poses for now * * *


### AUTOMATED CODE TO FACE SWAP FOR EVERY IMAGE!!!

# test = [3, 23, 51, 118]

for image_number in test:

    for pose_number in range(0, len(pose_types)):
    # for pose_number in range(1, 5):

        # We place code inside try/except block because some images have poor facial quality, meaning OpenCV cannot find landmarks
        try: 
            # Retrieve source image
            source_image = cv2.imread("/home/kate/Unselfie/detectron2/projects/DensePose/Resized_Images/Resized_Image" + str(image_number) + ".png")
            source_image_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

            Retrieve target image
            target_image = cv2.imread("/home/kate/Unselfie/UniHuman/code/additional_resized_results/Resized_Image" + str(image_number) + ".png_to_" + str(pose_types[pose_number]) + "_target.jpg.png")
            target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

            # # Retrieve ControlNet target image
            # target_image = cv2.imread("/home/kate/Unselfie/UniHuman/code/ControlNet_Outputs/Image" + str(image_number) + "_" + "ControlNet" + str(pose_number) + ".png")
            # target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

            # Creates a new mask
            mask = np.zeros((source_image.shape[0], source_image.shape[1]), dtype=source_image.dtype)

            # Creates destination image for reconstructing
            new_target_image = np.zeros_like(target_image)

            # Loading models and predictors of the dlib library to detect landmarks in both faces (specifically Face detector and Face landmarks detector)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("/home/kate/Unselfie/UniHuman/code/shape_predictor_68_face_landmarks.dat")

            # Getting landmarks FOR THE SOURCE IMAGE
            faces = detector(source_image_gray)
            for face in faces:
                landmarks = predictor(source_image_gray, face)
                landmarks_points = []

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points.append((x,y))

                # First converts regular Python array to NumPy array; then, it obtains the convex hull of the points; utilizes convex hull
                points = np.array(landmarks_points, np.int32)
                convexhull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, convexhull, 255)
                face_image_1 = cv2.bitwise_and(source_image, source_image, mask=mask)

                # Delaunay triangulation FOR THE SOURCE IMAGE
                rect = cv2.boundingRect(convexhull)
                subdiv = cv2.Subdiv2D(rect)
                subdiv.insert(landmarks_points)
                triangles = subdiv.getTriangleList()
                triangles = np.array(triangles, dtype=np.int32)

                indexes_triangles = []

                for t in triangles:
                    pt1 = (t[0], t[1])
                    pt2 = (t[2], t[3])
                    pt3 = (t[4], t[5])

                    index_pt1 = np.where((points == pt1).all(axis=1))
                    index_pt1 = extract_index_nparray(index_pt1)

                    index_pt2 = np.where((points == pt2).all(axis=1))
                    index_pt2 = extract_index_nparray(index_pt2)

                    index_pt3 = np.where((points == pt3).all(axis=1))
                    index_pt3 = extract_index_nparray(index_pt3)

                    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                        triangle = [index_pt1, index_pt2, index_pt3]
                        indexes_triangles.append(triangle)


            # Getting landmarks FOR THE TARGET IMAGE
            target_faces = detector(target_image_gray)
            placeholder_array = np.zeros_like(source_image)

            for face in target_faces:
                landmarks = predictor(target_image_gray, face)
                target_landmarks_points = []

                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    target_landmarks_points.append((x,y))

                placeholder_array = np.array(target_landmarks_points, np.int32)

                # Delaunay triangulation FOR BOTH IMAGES
                for triangle_index in indexes_triangles:
                    # Triangulation of the source image
                    tr1_pt1 = landmarks_points[triangle_index[0]]
                    tr1_pt2 = landmarks_points[triangle_index[1]]
                    tr1_pt3 = landmarks_points[triangle_index[2]]
                    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32) 

                    rect1 = cv2.boundingRect(triangle1)
                    (x, y, w, h) = rect1
                    cropped_triangle = source_image[y: y + h, x: x + w]
                    cropped_tr1_mask = np.zeros((h, w), np.uint8)

                    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
                    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)
                    
                    # Triangulation of second face
                    tr2_pt1 = target_landmarks_points[triangle_index[0]]
                    tr2_pt2 = target_landmarks_points[triangle_index[1]]
                    tr2_pt3 = target_landmarks_points[triangle_index[2]]
                    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                    rect2 = cv2.boundingRect(triangle2)
                    (x, y, w, h) = rect2
                    cropped_triangle2 = target_image[y: y + h, x: x + w]
                    cropped_tr2_mask = np.zeros((h, w), np.uint8)

                    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
                    cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

                    # Warp triangles
                    points = np.float32(points)
                    points2 = np.float32(points2)
                    M = cv2.getAffineTransform(points, points2)                 
                    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w,h), flags=cv2.INTER_NEAREST) # the "flags" parameter helps get rid of the black lines between triangles

                    # Reconstruct destination face PART 1
                    new_target_rect_area = new_target_image[y: y + h, x: x + w]
                    new_target_rect_area_gray = cv2.cvtColor(new_target_rect_area, cv2.COLOR_BGR2GRAY)

                    # Let's create a mask to remove the lines between the triangles --> FIX THIS *****
                    _, mask_triangles_designed = cv2.threshold(new_target_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                    # Reconstructing destination face PART 2
                    new_target_rect_area = cv2.add(new_target_rect_area, warped_triangle)
                    new_target_image[y: y + h, x: x + w] = new_target_rect_area

            target_points = placeholder_array.copy()
            convexhull2 = cv2.convexHull(target_points)

            # Face swapping procedure (putting source image onto target image)
            target_face_mask = np.zeros_like(target_image_gray)
            target_head_mask = cv2.fillConvexPoly(target_face_mask, convexhull2, 255)
            target_face_mask = cv2.bitwise_not(target_head_mask)

            target_head_noface = cv2.bitwise_and(target_image, target_image, mask=target_face_mask)
            result = cv2.add(target_head_noface, new_target_image)

            # Creates seamless clone to make resultant image appear more natural
            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
            seamlessclone = cv2.seamlessClone(result, target_image, target_head_mask, center_face2, cv2.NORMAL_CLONE) # ISN'T WORKING

            # cv2.imwrite("/home/kate/Unselfie/UniHuman/code/face_swapping_results/Image" + str(image_number) + "_" + str(pose_types[pose_number]) + "_FaceSwap.jpg", result)
            cv2.imwrite("/home/kate/Unselfie/UniHuman/code/seamless_clone_results/Image" + str(image_number) + "_" + str(pose_types[pose_number]) + "_FaceSwap.jpg", seamlessclone) # COMMENT THIS BACK IN FOR UNIHUMAN RESULTS
            # cv2.imwrite("/home/kate/Unselfie/UniHuman/code/ControlNet_FaceWarp/Image" + str(image_number) + "_ControlNet" + str(pose_number) + "_FaceSwap.jpg", seamlessclone) # USE THIS ONLY FOR CONTROLNET RESULTS

            print("Process complete for image" + str(image_number) + "_" + str(pose_types[pose_number]) + "! Yay!")

        except:
            print("Error for image" + str(image_number) + "_" + str(pose_types[pose_number]))

print("Overall process complete! Yay!")
