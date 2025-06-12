from qr_code_detector import QRCodeDetector
import os

def main():
    detector = QRCodeDetector()
    image_path = os.path.join(os.path.dirname(__file__), "test_media", "qr_code_5.png")

    try:
        qr_codes = detector.detect_from_image(image_path)
        if qr_codes:
            print(f"Fichier : {image_path}")
            for qr in qr_codes:
                print(f"Contenu du QR Code : {qr}")
        else:
            print(f"Aucun QR Code trouvé dans {image_path}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Erreur inattendue : {e}")

if __name__ == "__main__":
    main()