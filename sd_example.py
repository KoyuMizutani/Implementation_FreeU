from PIL import Image  
import torch
from diffusers import StableDiffusionPipeline
import warnings
warnings.filterwarnings("ignore")

model_id = "stabilityai/stable-diffusion-2-1"
pip = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip = pip.to("cuda")
seed = 42

torch.manual_seed(seed)
prompt = "A drone view of celebration with Christmas tree and fireworks, starry sky - background."
sd_image = pip(prompt, num_inference_steps=25).images[0]
sd_image.save("example.png")