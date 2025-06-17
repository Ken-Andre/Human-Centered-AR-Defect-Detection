
import shutil
from pathlib import Path

# ⚙️ Adapte ce chemin à TON dossier projet
base_path = Path("/dataset/screw")
classification_root = base_path.parent / "classification"

src_good = base_path / "train" / "good"
src_defect_root = base_path / "test"
dst_good = classification_root / "good"
dst_defect = classification_root / "defect"

# Création des dossiers (avec overwrite)
for d in [dst_good, dst_defect]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# 📦 Copie des vis "good"
for img in src_good.glob("*.png"):
    shutil.copy(img, dst_good / img.name)

# 📦 Copie des vis "defect"
for subfolder in src_defect_root.iterdir():
    if subfolder.is_dir() and subfolder.name != "good":
        for img in subfolder.glob("*.png"):
            shutil.copy(img, dst_defect / img.name)

print(f"[INFO] Copié {len(list(dst_good.glob('*.png')))} images dans 'good'")
print(f"[INFO] Copié {len(list(dst_defect.glob('*.png')))} images dans 'defect'")
