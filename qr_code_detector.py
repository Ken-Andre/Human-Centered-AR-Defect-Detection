import cv2
import os

class QRCodeDetector:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def detect_from_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Le fichier {image_path} est introuvable.")
        
        # Load the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image à partir de {image_path}")
        
        # Detect and decode QR codes
        data, points, _ = self.detector.detectAndDecode(image)
        if points is not None and data:
            return [data]  # Return a list with the decoded string
        else:
            return []  # Return an empty list if no QR code is found