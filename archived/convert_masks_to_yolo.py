import cv2
import os
import numpy as np
import shutil

# Reset label directory
label_dir = "../dataset/screw/labels"
if os.path.exists(label_dir):
    shutil.rmtree(label_dir)
    print(f"[INFO] Reset label folder: {label_dir}")

def mask_to_yolo_bbox(mask_path, img_width, img_height):
    """Convertit un masque en boîtes englobantes normalisées pour YOLO."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Normalisation pour YOLO (0 à 1)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        # Classe 0 pour "defect"
        bboxes.append(f"0 {x_center} {y_center} {width} {height}")
    return bboxes

def generate_annotations(test_dir, ground_truth_dir, label_dir):
    """Génère les fichiers d'annotations YOLO à partir des masques."""
    os.makedirs(label_dir, exist_ok=True)
    for defect_type in os.listdir(ground_truth_dir):
        gt_path = os.path.join(ground_truth_dir, defect_type)
        img_path = os.path.join(test_dir, defect_type)
        label_path = os.path.join(label_dir, defect_type)
        if os.path.isdir(gt_path):
            for mask_file in os.listdir(gt_path):
                if mask_file.endswith("_mask.png"):
                    mask_full_path = os.path.join(gt_path, mask_file)
                    img_name = mask_file.replace("_mask.png", ".png")
                    img_full_path = os.path.join(img_path, img_name)
                    if os.path.exists(img_full_path):
                        img = cv2.imread(img_full_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w = img.shape[:2]
                        bboxes = mask_to_yolo_bbox(mask_full_path, w, h)
                        if bboxes:
                            os.makedirs(os.path.dirname(os.path.join(label_path, img_name.replace(".png", ".txt"))), exist_ok=True)
                            with open(os.path.join(label_path, img_name.replace(".png", ".txt")), "w") as f:
                                for bbox in bboxes:
                                    f.write(bbox + "\n")

# Chemins (ajuste selon ton système)
test_dir = "../dataset/screw/test"
ground_truth_dir = "../dataset/screw/ground_truth"
label_dir = "../dataset/screw/labels"

# Exécuter la conversion
generate_annotations(test_dir, ground_truth_dir, label_dir)