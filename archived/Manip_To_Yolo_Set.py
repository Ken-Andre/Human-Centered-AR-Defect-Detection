import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Convertit tous les masques en labels YOLO
def mask_to_yolo_bbox(mask_path, img_width, img_height):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        bboxes.append(f"0 {x_center} {y_center} {width} {height}")
    return bboxes

def generate_annotations(test_dir, ground_truth_dir, label_dir):
    for defect_type in os.listdir(ground_truth_dir):
        gt_path = os.path.join(ground_truth_dir, defect_type)
        img_path = os.path.join(test_dir, defect_type)
        label_path = os.path.join(label_dir, defect_type)
        if os.path.isdir(gt_path):
            os.makedirs(label_path, exist_ok=True)
            for mask_file in os.listdir(gt_path):
                if mask_file.endswith("_mask.png"):
                    mask_full_path = os.path.join(gt_path, mask_file)
                    img_name = mask_file.replace("_mask.png", ".png")
                    img_full_path = os.path.join(img_path, img_name)
                    if os.path.exists(img_full_path):
                        img = cv2.imread(img_full_path)
                        h, w = img.shape[:2]
                        bboxes = mask_to_yolo_bbox(mask_full_path, w, h)
                        if bboxes:
                            with open(os.path.join(label_path, img_name.replace(".png", ".txt")), "w") as f:
                                for bbox in bboxes:
                                    f.write(bbox + "\n")

# 2. Génère les labels vides pour les good
def generate_empty_labels(good_img_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    for img_file in os.listdir(good_img_dir):
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            txt_name = img_file.rsplit(".", 1)[0] + ".txt"
            txt_path = os.path.join(output_label_dir, txt_name)
            open(txt_path, "w").close()

# 3. Réorganise tout dans la structure YOLO flat
def prepare_final_yolo_dataset(dataset_root="dataset/screw", output_root="dataset/screw/yolo"):
    # Supprime ancienne structure si existe
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(os.path.join(output_root, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "test/images"), exist_ok=True)
    os.makedirs(os.path.join(output_root, "test/labels"), exist_ok=True)

    # Collecte toutes les images + labels dispo (good + défauts) en tuples (path_img, path_label)
    image_label_pairs = []
    # good images
    good_img_dir = os.path.join(dataset_root, "train/good")
    good_label_dir = os.path.join(dataset_root, "labels/good")
    for img_file in os.listdir(good_img_dir):
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            img_path = os.path.join(good_img_dir, img_file)
            label_path = os.path.join(good_label_dir, img_file.replace(".png", ".txt").replace(".jpg", ".txt"))
            image_label_pairs.append((img_path, label_path))

    # défauts (test/<type>)
    defect_types = [d for d in os.listdir(os.path.join(dataset_root, "test")) if os.path.isdir(os.path.join(dataset_root, "test", d))]
    for defect_type in defect_types:
        defect_img_dir = os.path.join(dataset_root, "test", defect_type)
        defect_label_dir = os.path.join(dataset_root, "labels", defect_type)
        for img_file in os.listdir(defect_img_dir):
            if img_file.endswith(".png") or img_file.endswith(".jpg"):
                img_path = os.path.join(defect_img_dir, img_file)
                label_path = os.path.join(defect_label_dir, img_file.replace(".png", ".txt").replace(".jpg", ".txt"))
                image_label_pairs.append((img_path, label_path))

    # Split en train/test (80/20)
    train_pairs, test_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=42, shuffle=True)

    # Copie images/labels dans la bonne structure
    def copy_pairs(pairs, split):
        for img_path, label_path in pairs:
            img_name = os.path.basename(img_path)
            label_name = os.path.basename(label_path)
            dst_img = os.path.join(output_root, f"{split}/images", img_name)
            dst_label = os.path.join(output_root, f"{split}/labels", label_name)
            shutil.copy(img_path, dst_img)
            shutil.copy(label_path, dst_label)

    copy_pairs(train_pairs, "train")
    copy_pairs(test_pairs, "test")
    print(f"[INFO] {len(train_pairs)} images en train, {len(test_pairs)} images en test. Tout est à plat.")

# 4. Génère le dataset.yaml compatible YOLO
def generate_yolo_dataset_yaml(yolo_root="dataset/screw/yolo"):
    yaml_content = f"""
path: {yolo_root}
train: train/images
val: test/images
test: test/images

names:
  0: defect
"""
    yaml_path = os.path.join(yolo_root, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
    print(f"[INFO] Fichier dataset.yaml écrit: {yaml_path}")

if __name__ == "__main__":
    dataset_root = "dataset/screw"

    # (1) Génère les labels à partir des masques
    print("[STEP 1] Génération des labels YOLO pour les défauts")
    generate_annotations(
        test_dir=os.path.join(dataset_root, "test"),
        ground_truth_dir=os.path.join(dataset_root, "ground_truth"),
        label_dir=os.path.join(dataset_root, "labels"),
    )

    # (2) Génère les labels vides pour les 'good'
    print("[STEP 2] Génération des labels YOLO vides pour les 'good'")
    generate_empty_labels(
        good_img_dir=os.path.join(dataset_root, "train/good"),
        output_label_dir=os.path.join(dataset_root, "labels/good")
    )

    # (3) Création du dataset YOLO tout à plat
    print("[STEP 3] Fusion et flatten du dataset (train/test/images + labels)")
    prepare_final_yolo_dataset(dataset_root=dataset_root, output_root=os.path.join(dataset_root, "yolo"))

    # (4) Génère le fichier yaml
    print("[STEP 4] Génération du fichier dataset.yaml")
    generate_yolo_dataset_yaml(yolo_root=os.path.join(dataset_root, "yolo"))

    print("[FINI] Dataset YOLO prêt à l'emploi dans dataset/screw/yolo")
