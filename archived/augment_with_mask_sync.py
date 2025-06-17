import os
import cv2
import numpy as np
from glob import glob
import shutil

# Cleanup old augmentations
def clean_augmented(test_dir, mask_dir):
    for cls in os.listdir(test_dir):
        cls_path = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_path): continue
        for f in os.listdir(cls_path):
            if "aug" in f:
                os.remove(os.path.join(cls_path, f))

    for cls in os.listdir(mask_dir):
        cls_path = os.path.join(mask_dir, cls)
        if not os.path.isdir(cls_path): continue
        for f in os.listdir(cls_path):
            if "aug" in f:
                os.remove(os.path.join(cls_path, f))


def augment(img, mask):
    results = [(
        cv2.flip(img, 1),
        cv2.flip(mask, 1)
    ), (
        cv2.convertScaleAbs(img, alpha=1.2, beta=20),
        mask.copy()
    )]

    # Horizontal flip

    # Light contrast changes

    # Rotation
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), 10, 1)
    results.append((
        cv2.warpAffine(img, M, (img.shape[1], img.shape[0])),
        cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]))
    ))

    return results

def process_folder(img_dir, mask_dir):
    for file in sorted(glob(f"{mask_dir}/*_mask.png")):
        name = os.path.basename(file).replace("_mask.png", "")
        img_file = os.path.join(img_dir, name + ".png")
        if not os.path.exists(img_file):
            continue

        img = cv2.imread(img_file)
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        augmented = augment(img, mask)
        for i, (aug_img, aug_mask) in enumerate(augmented):
            cv2.imwrite(os.path.join(img_dir, f"{name}_aug{i}.png"), aug_img)
            cv2.imwrite(os.path.join(mask_dir, f"{name}_aug{i}_mask.png"), aug_mask)

# Exécution
clean_augmented("../dataset/screw/test", "dataset/screw/ground_truth")
# À appliquer
base = "dataset/screw/test"
gt = "dataset/screw/ground_truth"

for cls in os.listdir(gt):
    process_folder(os.path.join(base, cls), os.path.join(gt, cls))











