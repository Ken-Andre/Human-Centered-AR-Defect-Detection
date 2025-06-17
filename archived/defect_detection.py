from ultralytics import YOLO
import cv2

class DefectDetector:
    def __init__(self, model_path="models/best_screw.pt"):
        self.model = YOLO(model_path)  # Charge le modèle pré-entraîné ou entraîné

    def detect_defects(self, frame):
        results = self.model.predict(frame, conf=0.5)  # Ajuste le seuil de confiance
        defects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])  # Classe du défaut (ex. 0 pour "defect")
                conf = box.conf[0]     # Confiance
                defects.append({"class": cls, "confidence": float(conf), "bbox": box.xyxy[0].tolist()})
        return defects if defects else []

if __name__ == "__main__":
    detector = DefectDetector()
    frame = cv2.imread("test_media/defect_image.jpg")  # Test avec une image
    defects = detector.detect_defects(frame)
    print("Défauts détectés :", defects)