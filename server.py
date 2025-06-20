#!/usr/bin/env python3
"""
Serveur AR Assembly Detection (API, sans persistance capteurs/défauts)
- Flask + WebSocket pour communications AR/ESP32
- PostgreSQL pour gestion équipements SEULEMENT
- Intégration détection IA (Autoencoder, QR)
"""

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room, emit
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from datetime import datetime
import logging

import threading
import sys
import os

from sudo_detection import DetectionManager
from db import *
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'ar_assembly_detection_2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# -- "db_manager" qui wrappe uniquement l'accès aux équipements --
class DatabaseManager:
    def test_connection(self):
        try:
            # Connexion directe
            # from db import get_db_connection
            conn = get_db_connection()
            if conn is None:
                logger.error("No DB connection could be established.")
                return False
            # Test trivial
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"DB connection test failed: {e}")
            return False

    def get_equipment_details(self, serial_number):
        return get_equipment_details(serial_number)
    def save_sensor_data(self, serial_number, data):
        # Pas de persistance : on ne fait rien
        # TODO: implement future persistence
        return True
    def save_detection_result(self, serial_number, detection_result):
        # Pas de persistance : on ne fait rien
        # TODO: implement future persistence
        return True

def hot_reload_listener():
    """
    Thread qui écoute l'entrée clavier, si 'r' ou 'R' est pressé, redémarre le script.
    """
    import time
    while True:
        try:
            key = input("\nTape 'r' + Entrée pour redémarrer le serveur à chaud (hot-reload) : ")
            if key.strip().lower() == 'r':
                logger.info("[HOT-RELOAD] Restart demandé par l'utilisateur.")
                # Redémarrage du script courant avec les mêmes arguments
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except (EOFError, KeyboardInterrupt):
            break
        time.sleep(0.2)

db_manager = DatabaseManager()
analyzer = DetectionManager()

current_session = {
    'equipment_id': None,
    'serial_number': None,
    'detection_active': False,
    'last_frame': None
}

def decode_image_from_request(req):
    if 'image' not in req.files:
        return None, "Missing image file"
    file = req.files['image']
    image_bytes = file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image), None

# === ROUTES REST API ===

@app.route("/detect_defect", methods=["POST"])
def detect_defect_route():
    frame, error = decode_image_from_request(request)
    if error:
        return jsonify({"error": error}), 400
    result = analyzer.detect_defect(frame)
    return jsonify(result)

@app.route("/detect_qrcode", methods=["POST"])
def detect_qrcode_route():
    frame, error = decode_image_from_request(request)
    if error:
        return jsonify({"error": error}), 400
    result = analyzer.detect_qrcode(frame)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    try:
        db_status = db_manager.test_connection()
        ai_status = analyzer.test_models()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'ok' if db_status else 'error',
            'ai_models': 'ok' if ai_status else 'error',
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/sensors/<serial_number>', methods=['POST'])
def receive_sensor_data(serial_number):
    """Réception données capteurs, stream temps réel seulement"""
    try:
        data = request.get_json()
        required_fields = ['temperature', 'vibration', 'timestamp']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required sensor fields'}), 400
        # Diffusion temps réel, pas de persistance !
        # socketio.emit('sensor_data', {
        #     'serial_number': serial_number,
        #     'data': data
        # }, namespace='/sensors')
        room = f"sensor_{serial_number}"
        socketio.emit('sensor_data', {
            'serial_number': serial_number,
            'data': data
        }, room=room, namespace='/')
        logger.info(f"Sensor data streamed for {serial_number}: T={data['temperature']}°C, V={data['vibration']}")
        return jsonify({'status': 'success', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Error receiving sensor data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/equipment/<serial_number>', methods=['GET'])
def get_equipment_info(serial_number):
    try:
        equipment = db_manager.get_equipment_details(serial_number)
        if equipment:
            return jsonify(equipment)
        else:
            return jsonify({'error': 'Equipment not found'}), 404
    except Exception as e:
        logger.error(f"Error fetching equipment {serial_number}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/equipment/<serial_number>/documentation', methods=['GET'])
def get_equipment_documentation(serial_number):
    """
    Retourne la documentation technique d'un équipement selon son serial_number.
    """
    try:
        equipment = db_manager.get_equipment_details(serial_number)
        if equipment and 'documentation' in equipment:
            return jsonify({
                'serial_number': serial_number,
                'documentation': equipment['documentation']
            })
        elif equipment:
            return jsonify({'error': 'No documentation found for this equipment'}), 404
        else:
            return jsonify({'error': 'Equipment not found'}), 404
    except Exception as e:
        logger.error(f"Error fetching documentation for {serial_number}: {e}")
        return jsonify({'error': str(e)}), 500

# === WEBSOCKET HANDLERS ===

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected', 'session_id': request.sid})

@socketio.on('disconnect')
def handle_disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    global current_session
    current_session = {'equipment_id': None, 'serial_number': None, 'detection_active': False, 'last_frame': None}

@socketio.on('video_stream')
def handle_video_stream(data):
    try:
        frame_data = base64.b64decode(data['frame'])
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            emit('error', {'message': 'Failed to decode video frame'})
            return
        current_session['last_frame'] = frame
        if not current_session['equipment_id']:
            qr_result = analyzer.detect_qrcode(frame)
            if qr_result and qr_result.get('qr_code'):
                serial_number = qr_result['qr_code']
                equipment = db_manager.get_equipment_details(serial_number)
                if equipment:
                    current_session['equipment_id'] = equipment['serial_number']
                    current_session['serial_number'] = serial_number
                    emit('equipment_identified', {'equipment': equipment, 'qr_data': qr_result})
                    logger.info(f"Equipment identified: {serial_number}")
        emit('stream_status', {'status': 'received', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Error processing video stream: {e}")
        emit('error', {'message': f'Video processing error: {str(e)}'})

@socketio.on('capture_frame')
def handle_capture_frame(data):
    try:
        if current_session['last_frame'] is None:
            emit('error', {'message': 'No frame available for capture'})
            return
        frame = current_session['last_frame']
        detection_result = analyzer.detect_defect(frame)
        result = {
            'timestamp': datetime.now().isoformat(),
            'equipment_id': current_session['equipment_id'],
            'serial_number': current_session['serial_number'],
            'detection': detection_result,
            'session_id': request.sid
        }
        emit('detection_result', result)
        logger.info(f"Detection completed for {current_session['serial_number']}: {detection_result.get('status', 'unknown')}")
    except Exception as e:
        logger.error(f"Error during frame capture detection: {e}")
        emit('error', {'message': f'Detection error: {str(e)}'})

# Événement d'abonnement à un serial_number
@socketio.on('subscribe_sensor', namespace='/')
def subscribe_sensor(data):
    serial_number = data.get('serial_number')
    if not serial_number:
        emit('subscription_status', {
            'status': 'error',
            'message': 'Missing serial_number'
        }, to=request.sid)
        return

    # Ajouter le client à la room correspondante
    room = f"sensor_{serial_number}"
    join_room(room)
    emit('subscription_status', {
        'status': 'subscribed',
        'serial_number': serial_number,
        'message': f'Subscribed to sensor data for {serial_number}'
    }, to=request.sid)

# Événement de désabonnement
@socketio.on('unsubscribe_sensor', namespace='/')
def unsubscribe_sensor(data):
    serial_number = data.get('serial_number')
    if not serial_number:
        emit('subscription_status', {
            'status': 'error',
            'message': 'Missing serial_number'
        }, to=request.sid)
        return

    # Retirer le client de la room
    room = f"sensor_{serial_number}"
    leave_room(room)
    emit('subscription_status', {
        'status': 'unsubscribed',
        'serial_number': serial_number,
        'message': f'Unsubscribed from sensor data for {serial_number}'
    }, to=request.sid)

# --- Gestion erreurs ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@socketio.on_error_default
def default_error_handler(e):
    logger.error(f"WebSocket error: {e}")
    emit('error', {'message': 'WebSocket communication error'})

def initialize_server():
    logger.info("Initializing AR Assembly Detection Server...")
    return True

if __name__ == '__main__':
    # threading.Thread(target=hot_reload_listener, daemon=True).start()
    if initialize_server():
        logger.info("Starting AR Assembly Detection Server (HTTPS enabled)...")
        logger.info("Server endpoints:")
        logger.info("  - Main server: https://0.0.0.0:5000")
        logger.info("  - WebSocket: wss://0.0.0.0:5000")
        logger.info("  - Sensor data: POST /sensors/<serial_number>")
        logger.info("  - Equipment info: GET /equipment/<serial_number>")
        logger.info("  - Health check: GET /health")

        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            log_output=True,
            ssl_context=('cert.pem', 'key.pem')
        )
    else:
        logger.error("Server initialization failed. Exiting...")
        exit(1)

