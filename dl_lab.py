import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from PIL import Image
from model import BiSeNet 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


net = BiSeNet(n_classes=19)
net.load_state_dict(torch.load('79999_iter.pth', map_location=device))
net.to(device)
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def get_mask(img_path, classes):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size 
    img_resized = img.resize((512, 512), Image.BILINEAR)
    
    with torch.no_grad():
        tensor = transform(img_resized).unsqueeze(0).to(device)
        out = net(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0) 
        
    mask = np.zeros_like(parsing, dtype=np.uint8)
    for c in classes:
        mask[parsing == c] = 255
        
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

if __name__ == "__main__":
    os.makedirs("data/dl_lab", exist_ok=True)
    kernel = np.ones((15, 15), np.uint8)
    
    for i in tqdm(range(1000)):
        name = f"{i:06d}.png"
        src_path = f"data/id/{name}"
        a_path = f"data/makeup_image_A/{name}"
        b_path = f"data/makeup_image_B/{name}"
        
        if not (os.path.exists(src_path) and os.path.exists(a_path) and os.path.exists(b_path)):
            continue
            
        src = cv2.imread(src_path)
        makeup_a = cv2.imread(a_path)
        makeup_b = cv2.imread(b_path)
        

        eye_raw = get_mask(a_path, [4, 5])
        skin_a = get_mask(a_path, [1])
        eye_expanded = cv2.dilate(eye_raw, kernel, iterations=3)
        eye_final = cv2.bitwise_or(eye_raw, cv2.bitwise_and(eye_expanded, skin_a))
        

        lip_raw = get_mask(b_path, [12, 13])
        skin_b = get_mask(b_path, [1])
        lip_expanded = cv2.dilate(lip_raw, kernel, iterations=1)
        lip_final = cv2.bitwise_or(lip_raw, cv2.bitwise_and(lip_expanded, skin_b))
        

        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(float)
        makeup_a_lab = cv2.cvtColor(makeup_a, cv2.COLOR_BGR2LAB).astype(float)
        makeup_b_lab = cv2.cvtColor(makeup_b, cv2.COLOR_BGR2LAB).astype(float)


        if np.any(eye_final):
            blur = cv2.GaussianBlur(eye_final.astype(float), (15, 15), 0) / 255.0
            

            src_lab[:, :, 1] = (makeup_a_lab[:, :, 1] * blur) + (src_lab[:, :, 1] * (1.0 - blur))
            src_lab[:, :, 2] = (makeup_a_lab[:, :, 2] * blur) + (src_lab[:, :, 2] * (1.0 - blur))
            

            p_mask = blur * 0.5
            src_lab[:, :, 0] = (makeup_a_lab[:, :, 0] * p_mask) + (src_lab[:, :, 0] * (1.0 - p_mask))


        if np.any(lip_final):
            blur = cv2.GaussianBlur(lip_final.astype(float), (15, 15), 0) / 255.0
            

            src_lab[:, :, 1] = (makeup_b_lab[:, :, 1] * blur) + (src_lab[:, :, 1] * (1.0 - blur))
            src_lab[:, :, 2] = (makeup_b_lab[:, :, 2] * blur) + (src_lab[:, :, 2] * (1.0 - blur))
            

            p_mask = blur * 0.5
            src_lab[:, :, 0] = (makeup_b_lab[:, :, 0] * p_mask) + (src_lab[:, :, 0] * (1.0 - p_mask))
        
        res = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        cv2.imwrite(f"data/dl_lab/{i:06d}_composite_AB.jpg", res)