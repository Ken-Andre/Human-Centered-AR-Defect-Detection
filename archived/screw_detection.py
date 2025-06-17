#!/usr/bin/env python3
"""
Solution complète pour détection de défauts sur vis.
Objectif: 90% de précision minimum.
Avec logging détaillé et docstrings partout.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import logging

# ----- CONFIGURATION LOGGING -----
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s][%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)

class ScrewDataset(Dataset):
    """
    Dataset optimisé pour vis avec augmentation agressive.

    Args:
        root_dir (str): Dossier contenant 'good' et 'defect'.
        transform (callable, optional): Transformation supplémentaire.
        augment (bool): Si True, applique des augmentations d'images.

    Attributes:
        samples (list): Liste des tuples (chemin_image, label)
        aug_transform (albumentations.Compose): Transformations pour entraînement
        base_transform (albumentations.Compose): Transformations de base pour val/test
    """
    def __init__(self, root_dir, transform=None, augment=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        self.samples = []
        logging.debug("Initialisation du dataset à partir de %s", root_dir)

        # Collecte des images
        for class_name in ['good', 'defect']:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    label = 0 if class_name == 'good' else 1
                    self.samples.append((str(img_path), label))
            else:
                logging.warning("Le dossier %s n'existe pas !", class_dir)

        self.aug_transform = A.Compose([
            A.Resize(224, 224),
            A.RandomRotate90(p=0.8),
            A.Flip(p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.MedianBlur(blur_limit=5, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.CoarseDropout(max_holes=5, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.base_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        """Retourne le nombre d'échantillons."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Récupère une image et son label.

        Args:
            idx (int): Index de l'image.

        Returns:
            tuple: (image (Tensor), label (int))
        """
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            logging.error("Impossible de lire l'image : %s", img_path)
            raise FileNotFoundError(f"Image non trouvée: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augment:
            image = self.aug_transform(image=image)['image']
        else:
            image = self.base_transform(image=image)['image']

        return image, label

class OptimizedScrewClassifier(nn.Module):
    """
    Classificateur optimisé pour vis avec EfficientNet-B0 + attention.

    Args:
        num_classes (int): Nombre de classes de sortie.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)

        # Gèle les premières couches pour accélérer le fine-tuning
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass avec attention spatiale.

        Args:
            x (Tensor): Batch d'images.

        Returns:
            Tensor: Prédictions du modèle.
        """
        features = self.backbone.features(x)
        attention = self.spatial_attention(features)
        features = features * attention
        x = self.backbone.avgpool(features)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x

class FocalLoss(nn.Module):
    """
    Focal Loss pour gérer le déséquilibre des classes.

    Args:
        alpha (float): Coefficient alpha.
        gamma (float): Coefficient gamma.
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Calcule la focal loss.

        Args:
            inputs (Tensor): Logits du modèle.
            targets (Tensor): Labels réels.

        Returns:
            Tensor: Loss scalaire.
        """
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss

def train_model(data_dir, epochs=50, batch_size=16):
    """
    Entraîne le modèle de classification sur le dataset de vis.

    Args:
        data_dir (str): Dossier racine du dataset (doit contenir 'good' et 'defect').
        epochs (int): Nombre d'époques.
        batch_size (int): Taille du batch.

    Returns:
        nn.Module: Le modèle entraîné.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Appareil utilisé : {device}")

    # Chargement des datasets
    try:
        train_dataset = ScrewDataset(data_dir, augment=True)
        val_dataset = ScrewDataset(data_dir, augment=False)
        if len(train_dataset) == 0:
            logging.critical("Dataset d'entraînement vide !")
            raise Exception("Aucune image trouvée dans le dataset.")
    except Exception as e:
        logging.critical("Erreur de chargement du dataset.", exc_info=True)
        raise

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = OptimizedScrewClassifier().to(device)
    criterion = FocalLoss(alpha=2, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_accuracy = 0
    patience_counter = 0
    max_patience = 10

    logging.info("Début de l'entraînement du modèle...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        logging.info(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_screw_classifier.pt')
            patience_counter = 0
            logging.info(f"Nouveau meilleur modèle sauvegardé à {val_acc:.2f}% sur validation.")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            logging.warning(f"Early stopping après {epoch+1} époques.")
            break

    logging.info(f"Meilleure précision atteinte sur validation: {best_accuracy:.2f}%")
    return model

def test_model(model_path, test_dir):
    """
    Teste le modèle sur un jeu de données, affiche les métriques, sauvegarde la matrice de confusion.

    Args:
        model_path (str): Chemin du modèle entraîné.
        test_dir (str): Dossier contenant le dataset de test.

    Returns:
        bool: True si précision >= 90%, sinon False.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Chargement du modèle depuis {model_path} sur {device}")

    model = OptimizedScrewClassifier()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    except Exception as e:
        logging.critical("Erreur lors du chargement du modèle entraîné.", exc_info=True)
        return False
    model.eval()

    test_dataset = ScrewDataset(test_dir, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []

    logging.info("Début du test du modèle...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_predictions, target_names=['Good', 'Defect'])
    logging.info("\n" + report)

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Defect'], yticklabels=['Good', 'Defect'])
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig('confusion_matrix.png')
    plt.close()

    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision_defect = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall_defect = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0

    logging.info(f"Précision globale: {accuracy*100:.2f}%")
    logging.info(f"Précision défaut: {precision_defect*100:.2f}%")
    logging.info(f"Rappel défaut: {recall_defect*100:.2f}%")

    return accuracy >= 0.9

def predict_single_image(model_path, image_path):
    """
    Prédit la classe d'une image unique (GOOD ou DEFECT).

    Args:
        model_path (str): Chemin du modèle entraîné.
        image_path (str): Chemin de l'image à prédire.

    Returns:
        tuple: (str: label, float: score confiance)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedScrewClassifier()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    except Exception as e:
        logging.critical("Erreur lors du chargement du modèle pour prédiction.", exc_info=True)
        return "ERREUR", 0.0
    model.eval()

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image=image)['image'].unsqueeze(0).to(device)
    except Exception as e:
        logging.critical(f"Erreur lors de la lecture ou prétraitement de l'image {image_path}", exc_info=True)
        return "ERREUR", 0.0

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        result = 'DEFECT' if predicted.item() == 1 else 'GOOD'
        conf_score = confidence.item() * 100

        logging.info(f"Résultat: {result} | Confiance: {conf_score:.2f}%")
        return result, conf_score

if __name__ == "__main__":
    DATA_DIR = "../dataset/classification"  # Adapter au chemin réel

    logging.info("=== ENTRAÎNEMENT DU MODÈLE ===")
    try:
        model = train_model(DATA_DIR, epochs=50, batch_size=16)
    except Exception as e:
        logging.critical("Arrêt du script : erreur critique pendant l'entraînement.", exc_info=True)
        exit(1)

    logging.info("=== TEST DU MODÈLE ===")
    success = test_model('best_screw_classifier.pt', DATA_DIR)

    if success:
        logging.info("✅ OBJECTIF ATTEINT: Précision > 90%")
    else:
        logging.warning("❌ Objectif non atteint, réentraînement recommandé")

    # Pour tester une image
    # predict_single_image('best_screw_classifier.pt', 'test_image.jpg')
