import os
import shutil

# Reset label directory
label_dir = "../dataset/screw/labels"
if os.path.exists(label_dir):
    shutil.rmtree(label_dir)
    print(f"[INFO] Reset label folder: {label_dir}")

def generate_empty_labels(good_img_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    for img_file in os.listdir(good_img_dir):
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            txt_name = img_file.rsplit(".", 1)[0] + ".txt"
            txt_path = os.path.join(output_label_dir, txt_name)
            open(txt_path, "w").close()
            print(f"[INFO] Created: {txt_path}")

# Exemple d'utilisation
generate_empty_labels(
    good_img_dir="../dataset/screw/train/good",
    output_label_dir="../dataset/screw/labels/good"
)
