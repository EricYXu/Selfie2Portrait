### ABOUT THE PROGRAM: Obtains the inpainting result of various segmentation masks applied to a resized original image using the SDXL model!
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import PIL as Image

### HELPER FUNCTIONS AND CODE FOR INPAINTING!!!

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

# Prompts to use for inpainting!
short_prompt = "A person casually standing in a neutral position."
long_prompt =  "A person casually standing straight with their arms pointing down along the side of the body."
generator = torch.Generator(device="cuda").manual_seed(0)

### AUTOMATED PROCESS TO OBTAIN SDXL MODEL INPAINTING RESULTS FOR ALL 154 IMAGES!!!

for image_number in range(1,155):
  # Obtain the mask url's and resized image url!
  img_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Resized_Images/Resized_Image" + str(image_number) + ".png" #this is the image to overlay --> this should be a square version of the original image
  tight_mask_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Tight_Mask_Results/Tight_Mask_Image" + str(image_number) + ".png" 
  # side_rect_mask_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Side_Rect_Mask_Results/Side_Rect_Mask_Image" + str(image_number) + ".png" # --> UNUSED DUE TO POOR PERFORMANCE
  blocky_mask_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Blocky_Mask_Results/Blocky_Mask_Image" + str(image_number) + ".png" 

  # Load the resized original image and its corresponding masks!
  image = load_image(img_url)
  tight_mask_image = load_image(tight_mask_url)
  # side_rect_mask_image = load_image(side_rect_mask_url) # POOR PERFORMANCE
  blocky_mask_image = load_image(blocky_mask_url)

  # Obtain different inpainting outputs based on the mask and prompt used
  tight_mask_short_prompt_output = pipe(
    prompt=short_prompt,
    image=image,
    mask_image=tight_mask_image,
    guidance_scale=8.0,
    num_inference_steps=20,  # steps between 15 and 30 work well for us --> originally 20
    strength=0.99,  # make sure to use strength below 1.0 --> originally 0.99
    generator=generator,
  ).images[0]
  print("An inpainting task done! (1/4)")

  tight_mask_long_prompt_output = pipe(
    prompt=long_prompt,
    image=image,
    mask_image=tight_mask_image,
    guidance_scale=8.0,
    num_inference_steps=20,  # steps between 15 and 30 work well for us --> originally 20
    strength=0.99,  # make sure to use strength below 1.0 --> originally 0.99
    generator=generator,
  ).images[0]
  print("An inpainting task done! (2/4)")

  # side_rect_mask_short_prompt_output = pipe(
  #   prompt=short_prompt,
  #   image=image,
  #   mask_image=side_rect_mask_image,
  #   guidance_scale=8.0,
  #   num_inference_steps=20,  # steps between 15 and 30 work well for us --> originally 20
  #   strength=0.99,  # make sure to use strength below 1.0 --> originally 0.99
  #   generator=generator,
  # ).images[0]
  # print("An inpainting task done! (3/6)")

  # side_rect_mask_long_prompt_output = pipe(
  #   prompt=long_prompt,
  #   image=image,
  #   mask_image=side_rect_mask_image,
  #   guidance_scale=8.0,
  #   num_inference_steps=20,  # steps between 15 and 30 work well for us --> originally 20
  #   strength=0.99,  # make sure to use strength below 1.0 --> originally 0.99
  #   generator=generator,
  # ).images[0]
  # print("An inpainting task done! (4/6)")

  blocky_mask_short_prompt_output = pipe(
    prompt=short_prompt,
    image=image,
    mask_image=blocky_mask_image,
    guidance_scale=8.0,
    num_inference_steps=20,  # steps between 15 and 30 work well for us --> originally 20
    strength=0.99,  # make sure to use strength below 1.0 --> originally 0.99
    generator=generator,
  ).images[0]
  print("An inpainting task done! (3/4)")

  blocky_mask_long_prompt_output = pipe(
    prompt=long_prompt,
    image=image,
    mask_image=blocky_mask_image,
    guidance_scale=8.0,
    num_inference_steps=20,  # steps between 15 and 30 work well for us --> originally 20
    strength=0.99,  # make sure to use strength below 1.0 --> originally 0.99
    generator=generator,
  ).images[0]
  print("An inpainting task done! (4/4)")

  # Saves each inpainting output!
  tight_mask_short_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/SDXL_Inpainting_Results/Image" + str(image_number) + "_Tight_Mask_Short_Prompt.png") 
  tight_mask_long_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/SDXL_Inpainting_Results/Image" + str(image_number) + "_Tight_Mask_Long_Prompt.png") 
  # side_rect_mask_short_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/SDXL_Inpainting_Results/Image" + str(image_number) + "_Side_Rect_Mask_Short_Prompt.png") 
  # side_rect_mask_long_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/SDXL_Inpainting_Results/Image" + str(image_number) + "_Side_Rect_Mask_Long_Prompt.png") 
  blocky_mask_short_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/SDXL_Inpainting_Results/Image" + str(image_number) + "_Blocky_Mask_Short_Prompt.png") 
  blocky_mask_long_prompt_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/SDXL_Inpainting_Results/Image" + str(image_number) + "_Blocky_Mask_Long_Prompt.png") 

  print("Inpainting results for the specified image" + str(image_number) + " obtained!")

print("Automated process complete!")


