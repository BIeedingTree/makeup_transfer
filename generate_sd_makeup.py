import os
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
from datasets.ledits_utils.inversion_utils import inversion_forward_process, inversion_reverse_process

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
PROMPT_FILE = "prompt.txt"
DATA_DIR = "data/id"
OUT_DIR = "data/makeup_image"

def load_512(image_path, device="cuda"):
    """Loads an image, resizes to 512x512, and formats for Stable Diffusion."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.LANCZOS)
    

    image_tensor = transforms.ToTensor()(image)
    image_tensor = (image_tensor * 2.0) - 1.0
    return image_tensor.unsqueeze(0).to(device)

def image_grid(tensor):
    """Converts a Stable Diffusion output tensor back into a PIL Image."""

    tensor = (tensor / 2 + 0.5).clamp(0, 1)

    tensor = tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((tensor[0] * 255).astype(np.uint8))

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(MODEL_ID, subfolder="scheduler")

def invert_image(x0, prompt_src="human face", steps=100, cfg_scale=0.0):
    pipe.scheduler.set_timesteps(steps)
    
    with torch.autocast("cuda"), torch.inference_mode():
        w0 = (pipe.vae.encode(x0).latent_dist.mode() * 0.18215).to(torch.float16)
        
    wt, zs, wts = inversion_forward_process(
        pipe, w0, etas=1, prompt=prompt_src, cfg_scale=cfg_scale,
        prog_bar=False, num_inference_steps=steps
    )
    return zs, wts

def apply_prompt(zs, wts, prompt_tar, skip=50, cfg_scale=10.0):
    w0, _ = inversion_reverse_process(
        pipe, xT=wts[skip], etas=1, prompts=[prompt_tar], cfg_scales=[cfg_scale],
        prog_bar=False, zs=zs[skip:]
    )
    
    with torch.autocast("cuda"), torch.inference_mode():
        x0_dec = pipe.vae.decode(1 / 0.18215 * w0).sample
        
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]
        
    return image_grid(x0_dec)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    
    with open(PROMPT_FILE, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    for i in tqdm(range(1000), desc="Generating SD Makeup Variants"):
        in_path = os.path.join(DATA_DIR, f"{i:06d}.png")
        out_path = os.path.join(OUT_DIR, f"{i:06d}.png")
        
        if not os.path.exists(in_path):
            continue
            
        target_prompt = random.choice(prompts)
        x0 = load_512(in_path, device=DEVICE).to(torch.float16)
        
        zs, wts = invert_image(x0, prompt_src="human face", steps=100, cfg_scale=0)   
        res_img = apply_prompt(zs, wts, prompt_tar=target_prompt, skip=50, cfg_scale=10)   
        
        res_img.save(out_path)