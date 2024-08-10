### ABOUT THE PROGRAM: Obtains the FACE inpainting result of the FACE segementation mask applied to a resized original image using the SDXL model!
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import PIL as Image

### HELPER FUNCTIONS AND CODE FOR INPAINTING!!!

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

# Prompts to use for inpainting!
prompt = "human face in neutral position"
generator = torch.Generator(device="cuda").manual_seed(0)

### AUTOMATED PROCESS TO OBTAIN SDXL MODEL FACE INPAINTING RESULTS FOR ALL 154 IMAGES!!!

for image_number in range(1,155):
  # Obtain the mask url's and resized image url!
  img_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Resized_Images/Resized_Image" + str(image_number) + ".png" #this is the image to overlay --> this should be a square version of the original image
  face_mask_url = "/home/kate/Unselfie/detectron2/projects/DensePose/Face_Masks/Face_Mask_Image" + str(image_number) + ".png" 

  # Load the resized original image and its corresponding masks!
  image = load_image(img_url)
  face_mask_image = load_image(face_mask_url)

  # Obtain different inpainting outputs based on the mask and prompt used
  face_mask_output = pipe(
    prompt=prompt,
    image=image,
    mask_image=face_mask_image,
    guidance_scale=8.0,
    num_inference_steps=20,  # steps between 15 and 30 work well for us --> originally 20
    strength=0.15,  # LOW STRENGTH = preservation of pixels, HIGH STRENGTH = generation of new pixels
    generator=generator,
  ).images[0]
  
  # Saves each inpainting output!
  face_mask_output.save("/home/kate/Unselfie/detectron2/projects/DensePose/Facepaint_Results/Image" + str(image_number) + "_Facepaint.png") 

  print("Face inpainting results for the specified image" + str(image_number) + " obtained!")

print("Automated process complete!")


