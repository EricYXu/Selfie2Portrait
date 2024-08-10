### ABOUT THE PROGRAM: INPAINTS THE BACKGROUND OF A POSE-TRANSFERRED IMAGE USING STABLE DIFFUSION 2 MODEL!

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
prompt = ""
target_pose_type = ["male_thin", "male_wide", "female_thin", "female_wide"]
background_list = ["field", "house balcony", "front yard", "waterfall", "pool", "forest", "beach", "city", "living room", "room", "small city with river", "bedroom", "city", "forest", "living room", "dining room", "temple", "white wall", "forest", "city street", "night city street", "black wall", "living room", "park"]

### AUTOMATED PROCESS TO OBTAIN STABLE DIFFUSION 2 MODEL INPAINTING RESULTS FOR ALL 154 IMAGES!!!

for image_number in range(1, 26): # used to be 1 to 155

    # Conditional prompt for inpainting
    prompt = background_list[image_number - 1]

    for pose_number in range(0, len(target_pose_type)):

        # Obtain the mask url's and resized image url!
        img_url = "/home/kate/Unselfie/UniHuman/code/resized_results/Resized_Image" + str(image_number) + ".png_to_" + str(target_pose_type[pose_number]) + "_horizontal_target.jpg.png"
        background_mask_url = "/home/kate/Unselfie/UniHuman/code/Background_Masks/" + str(target_pose_type[pose_number]) + "_Background_Mask.jpg" 

        # Load the resized original image and its corresponding masks!
        image = load_image(img_url)
        background_mask_image = load_image(background_mask_url)

        #The mask structure is white for inpainting and black for keeping as is
        background_mask_prompt_output = pipe(prompt=prompt, image=image, mask_image=background_mask_image, strength=0.50).images[0] # you can add parameters to the pipe like strength and number of inference steps
        background_mask_prompt_output.save("/home/kate/Unselfie/UniHuman/code/StableDiffusion2InpaintedBackgroundResults/Image" + str(image_number) + "_" + str(target_pose_type[pose_number]) + "_Background_Inpainted.png")

        print("Inpainting results for the " + str(target_pose_type[pose_number]) + " pose obtained!")
    
    print("Process done for image" + str(image_number))

print("Automated process complete!")