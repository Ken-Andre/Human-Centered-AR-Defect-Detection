import os

def generate_dataset_yaml(base_path, output_path):
    content = f"""\
path: {base_path}/yolo_format
train: images/train
val: images/val
test: images/val

names:
  0: defect
"""
    yaml_path = os.path.join(base_path, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"[INFO] dataset.yaml written to: {yaml_path}")

# Exécution
generate_dataset_yaml("../dataset/screw", "dataset/screw/dataset.yaml")
