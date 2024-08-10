### ABOUT THE PROGRAM: Obtains the inpainting result of various segmentation masks applied to a resized original image using the Stable Diffusion 2 model!
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import torch
import PIL as Image

### HELPER FUNCTIONS AND CODE FOR INPAINTING!!!

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16 # NOTE: what does this variant parameter do? should I add it or remove it?
)

pipe.to("cuda")
very_short_prompt = "A person"
short_prompt = "A person casually standing in a neutral position." # Prompts for inpainting
long_prompt = "A person casually standing straight with their arms pointing down along the side of the body."

### AUTOMATED PROCESS TO OBTAIN STABLE DIFFUSION 2 MODEL INPAINTING RESULTS FOR ALL 154 IMAGES!!!

for image_number in range(1,155):
    # Obtain the mask url's and resized image url!
    img_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Resized_Images/Resized_Image" + str(image_number) + ".png" #this is the image to overlay --> this should be a square version of the original image
    tight_mask_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Tight_Mask_Results/Tight_Mask_Image" + str(image_number) + ".png" 
    # side_rect_mask_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Side_Rect_Mask_Results/Side_Rect_Mask_Image" + str(image_number) + ".png"  # <-- THIS MASK PERFORMS POORLY
    blocky_mask_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Blocky_Mask_Results/Blocky_Mask_Image" + str(image_number) + ".png" 

    # Load the resized original image and its corresponding masks!
    image = load_image(img_url)
    tight_mask_image = load_image(tight_mask_url)
    # side_rect_mask_image = load_image(side_rect_mask_url) # <-- THIS MASK PERFORMS POORLY!
    blocky_mask_image = load_image(blocky_mask_url)

    #The mask structure is white for inpainting and black for keeping as is
    tight_mask_short_prompt_output = pipe(prompt=short_prompt, image=image, mask_image=tight_mask_image).images[0]
    tight_mask_short_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/StableDiffusion2_Inpainting_Results/Image" + str(image_number) +"_Tight_Mask_Short_Prompt.png")
    print("An inpainting task done! (1/4)")

    tight_mask_long_prompt_output = pipe(prompt=long_prompt, image=image, mask_image=tight_mask_image).images[0]
    tight_mask_long_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/StableDiffusion2_Inpainting_Results/Image" + str(image_number) +"_Tight_Mask_Long_Prompt.png")
    print("An inpainting task done! (2/4)")

    # side_rect_mask_short_prompt_output = pipe(prompt=short_prompt, image=image, mask_image=side_rect_mask_image).images[0]
    # side_rect_mask_short_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/StableDiffusion2_Inpainting_Results/Side_Rect_Mask_Short_Prompt_Image" + str(image_number) + ".png")
    # print("An inpainting task done! (3/6)")

    # side_rect_mask_long_prompt_output = pipe(prompt=long_prompt, image=image, mask_image=side_rect_mask_image).images[0]
    # side_rect_mask_long_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/StableDiffusion2_Inpainting_Results/Side_Rect_Mask_Long_Prompt_Image" + str(image_number) + ".png")
    # print("An inpainting task done! (4/6)")

    blocky_mask_short_prompt_output = pipe(prompt=short_prompt, image=image, mask_image=blocky_mask_image).images[0]
    blocky_mask_short_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/StableDiffusion2_Inpainting_Results/Image" + str(image_number) +"_Blocky_Mask_Short_Prompt.png")
    print("An inpainting task done! (3/4)")

    blocky_mask_long_prompt_output = pipe(prompt=long_prompt, image=image, mask_image=blocky_mask_image).images[0]
    blocky_mask_long_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/StableDiffusion2_Inpainting_Results/Image" + str(image_number) +"_Blocky_Mask_Long_Prompt.png")
    print("An inpainting task done! (4/4)")

    print("Inpainting results for the specified image" + str(image_number) + " obtained!")

print("Automated process complete!")