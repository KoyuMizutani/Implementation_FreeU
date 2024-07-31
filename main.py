from PIL import Image  
import torch
from diffusers import StableDiffusionPipeline
from utils import register_free_upblock2d, register_free_crossattn_upblock2d

model_id = "stabilityai/stable-diffusion-2-1"
pip = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pip = pip.to("cuda")

def generate_images(prompt, seed, b1, b2, s1, s2):
    torch.manual_seed(seed)
    
    # FreeUを用いずに画像を生成
    register_free_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
    register_free_crossattn_upblock2d(pip, b1=1.0, b2=1.0, s1=1.0, s2=1.0)
    sd_image = pip(prompt, num_inference_steps=25).images[0]
    
    # FreeUを用いて画像を生成
    register_free_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s2)
    register_free_crossattn_upblock2d(pip, b1=b1, b2=b2, s1=s1, s2=s2)
    freeu_image = pip(prompt, num_inference_steps=25).images[0]
    
    return sd_image, freeu_image

# 使用例（プロンプトは適宜変更）
prompt = "A drone view of celebration with Christmas tree and fireworks, starry sky - background."
seed = 42
b1, b2, s1, s2 = 1.1, 1.2, 0.2, 0.2

sd_image, freeu_image = generate_images(prompt, seed, b1, b2, s1, s2)

# 画像を保存
sd_image.save("sd_image.png")
freeu_image.save("freeu_image.png")
