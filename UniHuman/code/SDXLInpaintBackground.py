### ABOUT THE PROGRAM: INPAINTS THE BACKGROUND OF A POSE-TRANSFERRED IMAGE USING SDXL MODEL!

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import PIL as Image

### HELPER FUNCTIONS AND CODE FOR INPAINTING

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
generator = torch.Generator(device="cuda").manual_seed(0)
prompt = ""
target_pose_type = ["male_thin", "male_wide", "female_thin", "female_wide"]
background_list = ["field", "house balcony", "front yard", "waterfall", "pool", "forest", "beach", "city", "living room", "room", "small city with river", "bedroom", "city", "forest", "living room", "dining room", "temple", "white wall", "forest", "city street", "night city street", "black wall", "living room", "park"]


### AUTOMATED PROCESS TO OBTAIN SDXL BACKGROUND INPAINTING RESULTS FOR ALL 154 IMAGES!!!

for image_number in range(1,26): # supposed to be 1 to 155
  for pose_number in range(0, len(target_pose_type)):

    # Conditional prompt for inpainting
    prompt = background_list[image_number - 1]

    # Obtain the mask url's and resized image url!
    img_url = "/home/kate/Unselfie/UniHuman/code/resized_results/Resized_Image" + str(image_number) + ".png_to_" + str(target_pose_type[pose_number]) + "_horizontal_target.jpg.png"
    background_mask_url = "/home/kate/Unselfie/UniHuman/code/Background_Masks/" + str(target_pose_type[pose_number]) + "_Background_Mask.jpg" 

    # Load the resized original image and its corresponding masks!
    image = load_image(img_url)
    background_mask_image = load_image(background_mask_url)

    # Obtain different inpainting outputs based on the mask and prompt used
    background_mask_prompt_output = pipe(
      prompt=prompt,
      image=image,
      mask_image=background_mask_image,
      guidance_scale=8.0,
      num_inference_steps=20,  # steps between 15 and 30 work well for us --> originally 20
      strength=0.075,  # make sure to use strength below 1.0 --> originally 0.99; HIGH STRENGTH = lots of new pixels, LOW STRENGTH = past pixels preserved
      generator=generator,
    ).images[0]

    print("A background inpainting task done!")

    # Saves each inpainting output!
    background_mask_prompt_output.save("/home/kate/Unselfie/UniHuman/code/SDXLInpaintedBackgroundResults/Image" + str(image_number) + "_" + str(target_pose_type[pose_number]) + "_Background_Inpainted.png") 

    print("Inpainting results for the specified image" + str(image_number) + " with the " + str(target_pose_type[pose_number]) + " obtained!")

  print("Process done for image" + str(image_number))

print("Automated process complete!")


