import os
from PIL import Image

ffhq_dir = "/DATA/wendizheng/Stable-Makeup/FFHQ/images1024x1024"  
output_dir = "data/id"
os.makedirs(output_dir, exist_ok=True)


image_paths = []
for root, _, files in os.walk(ffhq_dir):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(root, f))
            if len(image_paths) == 1000: 
                break
    if len(image_paths) == 1000: 
        break

image_paths = sorted(image_paths)[:1000] 

print(f"Found {len(image_paths)} images, starting processing...")

for i, img_path in enumerate(image_paths):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((512, 512))
        img.save(os.path.join(output_dir, f"{i:06d}.png"))
        
        if (i + 1) % 100 == 0: 
            print(f"Processed {i + 1}/{len(image_paths)}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        
print("1,000 images saved to data/id/")
