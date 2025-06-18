# sudo_detection.py

import os
import torch
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms

from qr_code_detector import QRCodeDetector
from db import get_equipment_details
from anomaly_autoencoder_pipeline import UNetAutoencoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/autoencoder_screw.pt"
THRESHOLD = 0.0033

class DetectionManager:
    def __init__(self, model_path=MODEL_PATH, threshold=THRESHOLD):
        self.threshold = threshold
        self.device = DEVICE
        self.model = UNetAutoencoder().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            print(f"[DetectionManager] Error loading model: {e}")
            self.model_loaded = False

    def test_models(self):
        """
        Vérifie si le modèle IA (autoencoder) est bien chargé et prêt à l'emploi.
        Retourne True si OK, False sinon.
        """
        return getattr(self, "model_loaded", False)

    def _preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def detect_defect(self, input_data):
        if isinstance(input_data, str) and os.path.isfile(input_data):
            image = Image.open(input_data).convert("RGB")
        elif isinstance(input_data, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
        else:
            return {"error": "Invalid input type"}

        input_tensor = self._preprocess_image(image)
        with torch.no_grad():
            reconstructed = self.model(input_tensor)
            loss = torch.mean((input_tensor - reconstructed) ** 2).item()
        # print(loss)
        status = "DEFECT" if loss < self.threshold else "GOOD"
        return {"anomaly_score": round(loss, 5), "status": status}

    @staticmethod
    def detect_qrcode(input_data):
        if isinstance(input_data, str) and os.path.isfile(input_data):
            frame = cv2.imread(input_data)
        elif isinstance(input_data, np.ndarray):
            frame = input_data
        else:
            return {"error": "Invalid input type"}

        qr_code = QRCodeDetector().detect_from_frame(frame)
        if qr_code:
            return {"qr_code": qr_code}
        return {"error": "QR code non détecté"}


# Test local
if __name__ == "__main__":
    test_path = "test_media/qr_code_6.png"
    # test_path = "input.jpg"
    detector = DetectionManager()
    # output = analyze_equipment(test_path)
    output = DetectionManager.detect_qrcode(test_path)
    # output = detector.detect_defect(test_path)
    print(output)
