from flask import Flask, request, jsonify
import cv2
import pyzbar.pyzbar as pyzbar
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")
server_ip = "192.168.1.100"  # À remplacer
port = "8080"

@app.route('/stream', methods=['POST'])
def process_stream():
    # Recevoir l'image du flux
    file = request.files['frame']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Détecter QR code
    decoded = pyzbar.decode(img)
    if decoded:
        data = decoded[0].data.decode()
        serial_number = data.split("Serial Number: ")[1] if "Serial Number" in data else None
        if serial_number:
            return jsonify({"status": "qr_detected", "url": f"http://{server_ip}:{port}/{serial_number}", "data": data})
    
    # Si pas de QR code, détecter équipement
    results = model(img)
    if results[0].boxes:  # Équipement détecté
        return jsonify({"status": "equipment_detected", "type": "screw_or_zipper"})
    
    return jsonify({"status": "nothing_detected"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)