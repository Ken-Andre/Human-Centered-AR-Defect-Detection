#!/usr/bin/env python3
"""
Pipeline complète détection d'anomalie par auto-encodeur convolutionnel.
Prépare le dataset, entraîne, prédit et visualise tout !
Copie-colle, adapte les chemins de base, et lance !
"""

import os
import shutil
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==== PARAMÈTRES ====
# Racine du dataset d'origine
RAW_ROOT = "dataset/screw"
# Racine du dataset AE (sera créé automatiquement)
AE_ROOT = "dataset/ae"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==== 1. PRÉPARATION DATASET POUR AE ====
def prepare_ae_dataset(raw_root=RAW_ROOT, ae_root=AE_ROOT, ratio_test=0.2):
    """
    Crée un dataset à plat compatible auto-encodeur :
    - train/good/
    - test/good/
    - test/defect/
    Copie automatiquement à partir de la structure YOLO ou raw.
    """
    print("\n[PREP] Préparation du dataset pour auto-encodeur...")

    # Clean up
    if os.path.exists(ae_root):
        shutil.rmtree(ae_root)
    os.makedirs(f"{ae_root}/train/good", exist_ok=True)
    os.makedirs(f"{ae_root}/test/good", exist_ok=True)
    os.makedirs(f"{ae_root}/test/defect", exist_ok=True)

    # 1. Récupère toutes les images “good” du train
    train_good = []
    path_train_good = os.path.join(raw_root, "train", "good")
    for f in os.listdir(path_train_good):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            train_good.append(os.path.join(path_train_good, f))

    # Split en train/test
    random.shuffle(train_good)
    n_test = int(ratio_test * len(train_good))
    test_good = train_good[:n_test]
    train_good = train_good[n_test:]

    # Copie train/good
    for f in train_good:
        shutil.copy(f, f"{ae_root}/train/good/{os.path.basename(f)}")
    # Copie test/good
    for f in test_good:
        shutil.copy(f, f"{ae_root}/test/good/{os.path.basename(f)}")

    # 2. Toutes les images "défectueuses" dans test/defect (parcourt test/* sauf good)
    path_test = os.path.join(raw_root, "test")
    for sub in os.listdir(path_test):
        sub_path = os.path.join(path_test, sub)
        if sub != "good" and os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy(os.path.join(sub_path, f), f"{ae_root}/test/defect/{sub}_{f}")

    print(f"[PREP] Fini. train/good={len(os.listdir(f'{ae_root}/train/good'))} | test/good={len(os.listdir(f'{ae_root}/test/good'))} | test/defect={len(os.listdir(f'{ae_root}/test/defect'))}")

# ==== 2. DATASET PyTorch ====
class ScrewDataset(Dataset):
    def __init__(self, root_dir, img_size=IMG_SIZE):
        self.img_paths = []
        if os.path.isdir(root_dir):
            for f in os.listdir(root_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(root_dir, f))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, os.path.basename(self.img_paths[idx])

# ==== 3. AUTO-ENCODEUR SIMPLE ====
# ==== UNet-Autoencoder pour meilleure localisation ====
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_block2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.up_block1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.final = nn.Conv2d(32, 3, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        bn = self.bottleneck(p2)
        up2 = self.up2(bn)
        cat2 = torch.cat([up2, d2], 1)
        up2b = self.up_block2(cat2)
        up1 = self.up1(up2b)
        cat1 = torch.cat([up1, d1], 1)
        up1b = self.up_block1(cat1)
        out = self.final(up1b)
        return self.sigmoid(out)


# ==== 4. TRAIN ====
def train_autoencoder(train_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, img_size=IMG_SIZE, device=DEVICE):
    ds = ScrewDataset(train_dir, img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = UNetAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        losses = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            output = model(imgs)
            loss = criterion(output, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[TRAIN] Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.4f}")
    torch.save(model.state_dict(), "./models/autoencoder_screw.pt")
    print("[TRAIN] Modèle entraîné et sauvegardé : autoencoder_screw.pt")
    return model

# ==== 5. TEST + VISU ====
def test_autoencoder(model_path, test_dir, out_visu="anomaly_visu", img_size=IMG_SIZE, threshold=None, device=DEVICE):
    model = UNetAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    sets = [("good", os.path.join(test_dir, "good")), ("defect", os.path.join(test_dir, "defect"))]
    criterion = nn.MSELoss(reduction='none')
    os.makedirs(out_visu, exist_ok=True)

    print(f"\n[TEST] Résultats pour chaque image test :")
    all_scores = []
    for set_name, set_dir in sets:
        if not os.path.exists(set_dir):
            continue
        ds = ScrewDataset(set_dir, img_size)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        for imgs, fnames in loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                recon = model(imgs)
            diff = (imgs - recon).abs()
            score = diff.mean().item()
            all_scores.append((fnames[0], set_name, score))

            # Génère une heatmap d'anomalie
            diff_map = diff[0].mean(0).cpu().numpy()
            diff_map_norm = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

            img_np = imgs[0].permute(1,2,0).cpu().numpy()
            recon_np = recon[0].permute(1,2,0).cpu().numpy()

            # Affichage et sauvegarde visu
            fig, axs = plt.subplots(1,4,figsize=(14,3))
            axs[0].imshow(img_np)
            axs[0].set_title(f"Original ({fnames[0]})")
            axs[1].imshow(recon_np)
            axs[1].set_title("Reconstruit")
            axs[2].imshow(diff_map_norm, cmap="hot")
            axs[2].set_title("Anomaly Map")
            axs[3].imshow(img_np)
            axs[3].imshow(diff_map_norm, cmap="hot", alpha=0.5)
            axs[3].set_title("Overlay")
            for ax in axs:
                ax.axis('off')
            out_path = os.path.join(out_visu, f"{set_name}_{fnames[0]}")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print(f"{set_name.upper()} | {fnames[0]} | score={score:.4f} | visu: {out_path}")

    # Suggest threshold
    print("\n[TEST] Résumé des scores :")
    scores_good = [x[2] for x in all_scores if x[1] == "good"]
    scores_defect = [x[2] for x in all_scores if x[1] == "defect"]
    print(f"Good (min/mean/max): {np.min(scores_good):.4f} / {np.mean(scores_good):.4f} / {np.max(scores_good):.4f}")
    if scores_defect:
        print(f"Defect (min/mean/max): {np.min(scores_defect):.4f} / {np.mean(scores_defect):.4f} / {np.max(scores_defect):.4f}")
        return all_scores
    else:
        print("Aucune image 'defect' en test.")
    print("\n[TEST] Pour différencier automatiquement, choisis un threshold entre max(good) et min(defect) si possible.")

def threshold_and_report(all_scores, out_csv="anomaly_scores.csv"):
    """
    Prend la liste des scores (nom, catégorie, score), calcule un threshold auto,
    exporte en csv et affiche la matrice de confusion.
    """
    import pandas as pd
    df = pd.DataFrame(all_scores, columns=["filename", "set", "score"])
    mean_good = df[df["set"]=="good"]["score"].mean()
    mean_defect = df[df["set"]=="defect"]["score"].mean()
    threshold = (mean_good + mean_defect) / 2
    print(f"\n[THRESH] Threshold choisi (moyenne good/defect): {threshold:.4f}")
    df["pred"] = df["score"] > threshold
    df["true"] = df["set"] == "defect"
    acc = (df["pred"] == df["true"]).mean()
    print(f"[THRESH] Précision totale: {acc*100:.1f}%")
    print("[THRESH] Matrice de confusion:")
    print(pd.crosstab(df["true"], df["pred"], rownames=["Vérité"], colnames=["Prédiction"], margins=True))
    df.to_csv(out_csv, index=False)
    print(f"[THRESH] Export des scores dans {out_csv}")
    return threshold


# ==== 6. PIPELINE PRINCIPALE ====
if __name__ == "__main__":
    print("=== AUTO-ENCODEUR DÉTECTION ANOMALIE PIPELINE ===")
    # 1. Préparation du dataset
    # prepare_ae_dataset(RAW_ROOT, AE_ROOT)
    # 2. Entraînement sur train/good
    # train_autoencoder(f"{AE_ROOT}/train/good")
    # 3. Test sur test/good et test/defect avec visu heatmap + scores
    all_scores = test_autoencoder("./models/autoencoder_screw.pt", f"{AE_ROOT}/test")

    threshold_and_report(all_scores)
    print("\n=== TOUT EST FINI ===")
    print("=> Visualise les heatmaps dans ./anomaly_visu/")
