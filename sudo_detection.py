# sudo_detection.py

import os
import torch
from PIL import Image
from torchvision import transforms

from qr_code_detector import QRCodeDetector
from db import get_equipment_details
from anomaly_autoencoder_pipeline import UNetAutoencoder

# Détection sur image statique
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
THRESHOLD = 0.08  # Seuil calibré à adapter

# Chargement du modèle autoencodeur
MODEL_PATH = os.path.join("models", "autoencoder_screw.pt")
autoencoder = UNetAutoencoder().to(DEVICE)
autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
autoencoder.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Initialiser l’instance de QRCodeDetector
qr_detector = QRCodeDetector()

def detect_qr(image_path: str) -> dict:
    """Détecte un QR code dans une image et retourne les infos associées."""
    qr_codes = qr_detector.detect_from_image(image_path)
    if not qr_codes:
        return {"error": "QR code non détecté"}

    serial = qr_codes[0]  # On suppose un seul QR code
    equipment_info = get_equipment_details(serial)
    if not equipment_info:
        print("Qr-code content:", serial)
        return {"error": "Équipement inconnu dans la base"}

    return {"qr_code": serial, "equipment_info": equipment_info}

def detect_defect(image_path: str) -> dict:
    """Évalue si une vis est défectueuse (via reconstruction)."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return {"error": "Image invalide ou non trouvée"}

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon = autoencoder(input_tensor)
        diff = (input_tensor - recon).abs()
        score = diff.mean().item()

    is_defective = score > THRESHOLD
    return {
        "anomaly_score": round(score, 4),
        "status": "DEFECT" if is_defective else "GOOD"
    }

def analyze_equipment(image_path: str) -> dict:
    """Pipeline complet : QR + DB + détection autoencodeur."""
    result = detect_qr(image_path)
    if "error" in result:
        return result

    defect_result = detect_defect(image_path)
    return {
        **result,
        **defect_result
    }

# Test local
if __name__ == "__main__":
    test_path = "test_media/qr_code_6.png"
    # test_path = "input.jpg"
    # output = analyze_equipment(test_path)
    output = detect_qr(test_path)
    # output = detect_defect(test_path)
    print(output)
