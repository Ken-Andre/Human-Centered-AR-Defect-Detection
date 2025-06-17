import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
from ultralytics import YOLO
import cv2
import os
import sys
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"


# === Configuration ===
# CLASSIFIER_MODEL = "models/mobilenet_screw_classifier.pt"
CLASSIFIER_MODEL = "models/best_screw_classifier.pt"
YOLO_MODEL = "runs/detect/train6/best.pt"  # modèle YOLO entraîné
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
TEMP_PATH = "temp.jpg"  # YOLO nécessite cv2

# === Chargement du classifieur ===
classifier = mobilenet_v3_small(pretrained=False)
classifier.classifier[3] = nn.Linear(classifier.classifier[3].in_features, 2)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL, map_location="cpu"))
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === Étape 1 : Classification ===
def classify(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    output = classifier(input_tensor)
    _, pred = torch.max(output, 1)
    label = "good" if pred.item() == 0 else "defect"
    return label

# === Étape 2 : Détection (si nécessaire) ===
def detect_defect(image_path):
    model = YOLO(YOLO_MODEL)
    img = cv2.imread(image_path)
    results = model.predict(img, save=True, conf=0.25)
    print("[YOLO] Détection effectuée.")
    return results

# === Pipeline principal ===
result = classify(IMG_PATH)
print(f"[Classifier] Résultat : {result}")

if result == "defect":
    print("[Pipeline] Anomalie détectée. Lancement de la détection YOLO...")
    detect_defect(IMG_PATH)
else:
    print("[Pipeline] Aucun défaut détecté. Aucune détection YOLO nécessaire.")
