import os
import shutil

def reset_dataset(clean_src, working_dst):
    if os.path.exists(working_dst):
        shutil.rmtree(working_dst)
        print(f"[INFO] Removed existing dataset at {working_dst}")
    shutil.copytree(clean_src, working_dst)
    print(f"[INFO] Copied clean dataset from {clean_src} to {working_dst}")

# Utilisation
clean_path = r"C:\Users\yoann\Documents\School\X4\Recherche\screw"
target_path = r"dataset\screw"

reset_dataset(clean_src=clean_path, working_dst=target_path)
