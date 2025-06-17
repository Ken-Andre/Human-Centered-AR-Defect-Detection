#!/usr/bin/env python3
"""
Solution YOLO optimisée pour détection précise des défauts sur vis
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
import shutil
from ultralytics import YOLO
import torch

class OptimizedYOLOPreprocessor:
    """Préprocessing spécialisé pour améliorer la détection YOLO"""

    @staticmethod
    def enhance_defect_visibility(image_path, output_path):
        """Améliore la visibilité des défauts avant détection YOLO"""
        img = cv2.imread(image_path)

        # Conversion en niveaux de gris pour mieux voir les défauts
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE pour améliorer le contraste local
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Détection de contours pour révéler les défauts
        edges = cv2.Canny(enhanced, 50, 150)

        # Morphologie pour connecter les défauts fragmentés
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Fusion avec l'image originale
        enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Combinaison pondérée
        result = cv2.addWeighted(enhanced_3ch, 0.7, edges_3ch, 0.3, 0)

        cv2.imwrite(output_path, result)
        return output_path

def create_optimized_yolo_config():
    """Génère configuration YOLO optimisée pour défauts"""

    config = {
        'task': 'detect',
        'mode': 'train',
        'model': 'yolov8n.pt',
        'data': 'dataset/screw/yolo/dataset.yaml',
        'epochs': 300,
        'patience': 30,
        'batch': 4,  # Réduit pour CPU
        'imgsz': 640,
        'save': True,
        'save_period': 10,
        'cache': False,
        'device': 'cpu',
        'workers': 0,
        'project': 'runs/detect',
        'name': 'optimized_screw',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 20,
        'resume': False,
        'amp': False,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': True,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.1,
        'val': True,
        'split': 'val',
        'save_json': True,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
        'source': None,
        'vid_stride': 1,
        'stream_buffer': False,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'classes': None,
        'retina_masks': False,
        'embed': None,
        'show': False,
        'save_frames': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None,

        # Hyperparamètres optimisés pour défauts
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.1,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.1,
        'scale': 0.9,
        'shear': 2.0,
        'perspective': 0.0,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 0.8,
        'mixup': 0.1,
        'copy_paste': 0.3,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
    }

    with open('optimized_yolo_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return 'optimized_yolo_config.yaml'

def enhance_training_data(data_dir="dataset/screw/yolo"):
    """Améliore les données d'entraînement pour YOLO"""

    data_path = Path(data_dir)
    enhanced_path = data_path.parent / "screw_enhanced"

    # Créer structure enhanced
    for split in ['train', 'test']:
        for subfolder in ['images', 'labels']:
            (enhanced_path / split / subfolder).mkdir(parents=True, exist_ok=True)

    preprocessor = OptimizedYOLOPreprocessor()

    # Traiter les images d'entraînement
    for split in ['train', 'test']:
        images_dir = data_path / split / 'images'
        labels_dir = data_path / split / 'labels'

        if images_dir.exists():
            for img_file in images_dir.glob('*.png'):
                # Image enhanced
                enhanced_img_path = enhanced_path / split / 'images' / img_file.name
                preprocessor.enhance_defect_visibility(str(img_file), str(enhanced_img_path))

                # Copier le label correspondant
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    shutil.copy(label_file, enhanced_path / split / 'labels' / label_file.name)

    # Créer dataset.yaml pour enhanced
    dataset_config = {
        'path': str(enhanced_path.absolute()),
        'train': 'train/images',
        'val': 'test/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['defect']
    }

    with open(enhanced_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    return str(enhanced_path / 'dataset.yaml')

def train_optimized_yolo(data_yaml):
    """Entraîne YOLO avec configuration optimisée"""

    # Créer config optimisée
    config_path = create_optimized_yolo_config()

    # Modifier pour utiliser les données enhanced
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['data'] = data_yaml

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Lancer l'entraînement
    model = YOLO('../models/yolov8n.pt')

    # Entraînement avec validation continue
    results = model.train(
        data=data_yaml,
        epochs=200,
        imgsz=640,
        batch=4,
        device='cpu',
        workers=0,
        cache=False,
        patience=30,
        save=True,
        project='runs/detect',
        name='optimized_screw',
        exist_ok=True,

        # Hyperparamètres optimisés
        lr0=0.001,
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Augmentation adaptée aux défauts
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.9,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.3,

        # Early stopping
        cos_lr=True,
        close_mosaic=20,
    )

    # Sauvegarder le meilleur modèle
    best_model_path = '../runs/detect/optimized_screw/weights/best.pt'
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, '../models/optimized_yolo_screw.pt')
        print("✅ Modèle YOLO optimisé sauvé: models/optimized_yolo_screw.pt")

    return results

def test_yolo_precision(model_path, test_data):
    """Test précision YOLO avec métriques détaillées"""

    model = YOLO(model_path)

    # Validation avec métriques
    results = model.val(data=test_data, save_json=True, plots=True)

    print(f"\n=== RÉSULTATS YOLO ===")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Précision: {results.box.mp:.4f}")
    print(f"Rappel: {results.box.mr:.4f}")

    # Objectif: mAP50 > 0.90
    success = results.box.map50 > 0.90

    if success:
        print("✅ OBJECTIF YOLO ATTEINT: mAP50 > 90%")
    else:
        print("❌ Objectif YOLO non atteint")
        print("💡 Suggestions:")
        print("   - Augmenter epochs à 300")
        print("   - Vérifier qualité annotations")
        print("   - Utiliser YOLOv8s au lieu de YOLOv8n")

    return success

def integrated_cascade_pipeline(image_path):
    """Pipeline intégré: Classification + YOLO optimisé"""

    # Import du classificateur
    import torch
    from torchvision import transforms
    from PIL import Image
    from torchvision.models import efficientnet_b0
    import torch.nn as nn

    # Charger classificateur
    class OptimizedScrewClassifier(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.backbone = efficientnet_b0(weights='IMAGENET1K_V1')
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

        def forward(self, x):
            return self.backbone(x)

    # Charger modèles
    classifier = OptimizedScrewClassifier()
    classifier.load_state_dict(torch.load('../models/best_screw_classifier.pt', map_location='cpu'))
    classifier.eval()

    yolo_model = YOLO('../models/optimized_yolo_screw.pt')

    # Classification
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = classifier(input_tensor)
        _, pred = torch.max(output, 1)
        classification = 'DEFECT' if pred.item() == 1 else 'GOOD'

    print(f"Classification: {classification}")

    # Si défaut détecté, localiser avec YOLO
    if classification == 'DEFECT':
        print("Localisation des défauts...")

        # Préprocessing pour YOLO
        preprocessor = OptimizedYOLOPreprocessor()
        enhanced_path = 'temp_enhanced.jpg'
        preprocessor.enhance_defect_visibility(image_path, enhanced_path)

        # Détection YOLO
        results = yolo_model.predict(enhanced_path, conf=0.25, save=True, show_labels=True)

        # Compter détections
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"Défauts localisés: {detections}")

        # Nettoyer fichier temporaire
        if os.path.exists(enhanced_path):
            os.remove(enhanced_path)

        return classification, detections
    else:
        return classification, 0

# Fonction principale
def main():
    print("=== OPTIMISATION YOLO POUR DÉFAUTS VIS ===")

    # 1. Améliorer données d'entraînement
    print("1. Amélioration des données...")
    enhanced_data_yaml = enhance_training_data()
    data_yaml = "dataset/screw/yolo/dataset.yaml"

    # 2. Entraîner YOLO optimisé
    print("2. Entraînement YOLO optimisé...")
    train_optimized_yolo(enhanced_data_yaml)

    # 3. Tester précision
    print("3. Test de précision...")
    test_yolo_precision('../models/optimized_yolo_screw.pt', enhanced_data_yaml)

    print("\n=== PIPELINE INTÉGRÉ PRÊT ===")
    print("Utilisation:")
    print("integrated_cascade_pipeline('image.jpg')")

if __name__ == "__main__":
    main()