from PIL import Image
from pyzbar.pyzbar import decode
import os

class QRCodeDetector:
    def __init__(self):
        pass  # No initialization needed for Pyzbar

    def detect_from_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Le fichier {image_path} est introuvable.")

        # Load the image using PIL
        try:
            img = Image.open(image_path)
        except Exception as e:
            raise ValueError(f"Impossible de charger l'image à partir de {image_path}: {e}")

        # Decode the QR code
        results = decode(img)

        # If a QR code is found
        if results:
            # Return the decoded data as a list
            return [result.data.decode('utf-8') for result in results]
        else:
            print(f"Aucun code QR détecté dans {image_path}")
            return []

    def detect_from_frame(self, frame):
        # Convert OpenCV frame (numpy array) to PIL Image for Pyzbar
        from io import BytesIO
        from cv2 import cvtColor, COLOR_BGR2RGB
        img = Image.fromarray(cvtColor(frame, COLOR_BGR2RGB))
        results = decode(img)
        if results:
            return [result.data.decode('utf-8') for result in results]
        return []