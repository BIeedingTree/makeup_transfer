import cv2
import numpy as np
import dlib
import os
from tqdm import tqdm


ID_DIR = "data/id"
MAKEUP_A_DIR = "data/makeup_image_A"
MAKEUP_B_DIR = "data/makeup_image_B"
OUT_DIR = "data/partial_transfer"


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_dilated_mask(image_shape, points):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for i in tqdm(range(1000), desc="Compositing Eyes (A) + Lips (B)"):
        img_name = f"{i:06d}.png"
        
        src_path = os.path.join(ID_DIR, img_name)
        a_path = os.path.join(MAKEUP_A_DIR, img_name)
        b_path = os.path.join(MAKEUP_B_DIR, img_name)
        
        if not (os.path.exists(src_path) and os.path.exists(a_path) and os.path.exists(b_path)):
            continue
            
        src = cv2.imread(src_path)
        makeup_a = cv2.imread(a_path)
        makeup_b = cv2.imread(b_path)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if not faces:
            continue
            

        shape = predictor(gray, faces[0])
        coords = np.array([[shape.part(j).x, shape.part(j).y] for j in range(68)])
        eye_pts = coords[36:48]
        lip_pts = coords[48:68]

        eye_mask = get_dilated_mask(src.shape, eye_pts)
        lip_mask = get_dilated_mask(src.shape, lip_pts)
        

        eye_center = tuple(np.mean(eye_pts, axis=0).astype(int))
        lip_center = tuple(np.mean(lip_pts, axis=0).astype(int))
        

        step1 = cv2.seamlessClone(makeup_a, src, eye_mask, eye_center, cv2.NORMAL_CLONE)
        final_composite = cv2.seamlessClone(makeup_b, step1, lip_mask, lip_center, cv2.NORMAL_CLONE)
        

        out_path = os.path.join(OUT_DIR, f"{i:06d}_composite_AB.jpg")
        cv2.imwrite(out_path, final_composite)