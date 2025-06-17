import cv2
import os
import numpy as np

def augment_image(img):
    aug_list = [img, cv2.flip(img, 1), cv2.flip(img, 0), cv2.convertScaleAbs(img, alpha=1.2, beta=30),
                cv2.convertScaleAbs(img, alpha=0.8, beta=-30)]

    # Contraste et luminosité

    # Bruit
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    aug_list.append(cv2.add(img, noise))

    return aug_list

def augment_defect_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            aug_imgs = augment_image(img)
            base = file.rsplit(".", 1)[0]
            for i, aug in enumerate(aug_imgs):
                cv2.imwrite(os.path.join(folder_path, f"{base}_aug{i}.png"), aug)

# Appliquer à chaque type de défaut
root = "dataset/screw/test"
for subdir in os.listdir(root):
    if subdir != "good":
        print(f"[+] Augmenting: {subdir}")
        augment_defect_folder(os.path.join(root, subdir))
