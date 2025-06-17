import os
import shutil
from sklearn.model_selection import train_test_split
import shutil


def copy_with_labels(img_src, label_src, img_dst, label_dst):
    os.makedirs(os.path.dirname(img_dst), exist_ok=True)
    shutil.copy(img_src, img_dst)
    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)

def prepare_balanced_dataset(test_dir, label_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    all_images = []
    for cls_name in os.listdir(test_dir):
        cls_dir = os.path.join(test_dir, cls_name)
        if os.path.isdir(cls_dir):
            for img_file in os.listdir(cls_dir):
                if img_file.endswith(".png"):
                    all_images.append((cls_name, img_file))

    train_items, val_items = train_test_split(all_images, test_size=0.2, stratify=[x[0] for x in all_images])

    for split_name, items in [("train", train_items), ("val", val_items)]:
        for cls_name, img_file in items:
            img_src = os.path.join(test_dir, cls_name, img_file)
            label_src = os.path.join(label_dir, cls_name, img_file.replace(".png", ".txt"))
            img_dst = os.path.join(output_dir, "images", split_name, img_file)
            label_dst = os.path.join(output_dir, "labels", split_name, img_file.replace(".png", ".txt"))
            copy_with_labels(img_src, label_src, img_dst, label_dst)

output_dir= "../dataset/screw/yolo_format"
# Reset YOLO-format output
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"[INFO] Reset output YOLO dataset: {output_dir}")
# Lancement
prepare_balanced_dataset(
    test_dir="../dataset/screw/test",
    label_dir="../dataset/screw/labels",
    output_dir="../dataset/screw/yolo_format"
)
