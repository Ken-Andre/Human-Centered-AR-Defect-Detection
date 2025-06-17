#!/usr/bin/env python3
"""
Serveur principal pour le système AR Assembly Detection
- Flask + WebSocket pour communications AR
- PostgreSQL pour données équipements
- Intégration modules de détection (QR + IA)
- Réception données capteurs ESP32
"""

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import psycopg2
import json
import base64
import cv2
import numpy as np
import logging
from datetime import datetime
import os

# Imports modules de détection
from detection_integree import DetectionManager
from db import DatabaseManager

# Configuration logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ar_assembly_detection_2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialisation des managers
db_manager = DatabaseManager()
detection_manager = DetectionManager()

# Variables globales pour la session
current_session = {
    'equipment_id': None,
    'serial_number': None,
    'detection_active': False,
    'last_frame': None
}

# ==== ROUTES REST API ====

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification santé du serveur"""
    try:
        # Test connexion DB
        db_status = db_manager.test_connection()
        # Test modèles IA
        ai_status = detection_manager.test_models()

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
    """Réception données capteurs ESP32"""
    try:
        data = request.get_json()

        # Validation des données
        required_fields = ['temperature', 'vibration', 'timestamp']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required sensor fields'}), 400

        # Stockage en base
        success = db_manager.save_sensor_data(serial_number, data)

        if success:
            # Diffusion temps réel via WebSocket
            socketio.emit('sensor_data', {
                'serial_number': serial_number,
                'data': data
            }, namespace='/sensors')

            logger.info(f"Sensor data received for {serial_number}: T={data['temperature']}°C, V={data['vibration']}")
            return jsonify({'status': 'success', 'timestamp': datetime.now().isoformat()})
        else:
            return jsonify({'error': 'Failed to save sensor data'}), 500

    except Exception as e:
        logger.error(f"Error receiving sensor data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/equipment/<serial_number>', methods=['GET'])
def get_equipment_info(serial_number):
    """Récupération infos équipement"""
    try:
        equipment = db_manager.get_equipment_details(serial_number)
        if equipment:
            return jsonify(equipment)
        else:
            return jsonify({'error': 'Equipment not found'}), 404
    except Exception as e:
        logger.error(f"Error fetching equipment {serial_number}: {e}")
        return jsonify({'error': str(e)}), 500

# ==== WEBSOCKET HANDLERS ====

@socketio.on('connect')
def handle_connect():
    """Connexion client AR"""
    logger.info(f"AR Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected', 'session_id': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Déconnexion client AR"""
    logger.info(f"AR Client disconnected: {request.sid}")
    # Reset session
    global current_session
    current_session = {
        'equipment_id': None,
        'serial_number': None,
        'detection_active': False,
        'last_frame': None
    }

@socketio.on('video_stream')
def handle_video_stream(data):
    """Réception flux vidéo continu AR"""
    try:
        # Decode frame
        frame_data = base64.b64decode(data['frame'])
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'message': 'Failed to decode video frame'})
            return

        # Stockage frame courante
        current_session['last_frame'] = frame

        # Détection QR automatique si pas d'équipement identifié
        if not current_session['equipment_id']:
            qr_result = detection_manager.detect_qr_code(frame)
            if qr_result:
                serial_number = qr_result.get('serial_number')
                if serial_number:
                    equipment = db_manager.get_equipment_details(serial_number)
                    if equipment:
                        current_session['equipment_id'] = equipment['id']
                        current_session['serial_number'] = serial_number

                        emit('equipment_identified', {
                            'equipment': equipment,
                            'qr_data': qr_result
                        })
                        logger.info(f"Equipment identified: {serial_number}")

        # Confirmation réception stream
        emit('stream_status', {'status': 'received', 'timestamp': datetime.now().isoformat()})

    except Exception as e:
        logger.error(f"Error processing video stream: {e}")
        emit('error', {'message': f'Video processing error: {str(e)}'})

@socketio.on('capture_frame')
def handle_capture_frame(data):
    """Capture frame pour détection (déclenchée par commande vocale)"""
    try:
        if current_session['last_frame'] is None:
            emit('error', {'message': 'No frame available for capture'})
            return

        frame = current_session['last_frame']

        # Détection sur la frame capturée
        detection_result = detection_manager.detect_defects(frame)

        # Préparation résultat pour AR
        result = {
            'timestamp': datetime.now().isoformat(),
            'equipment_id': current_session['equipment_id'],
            'serial_number': current_session['serial_number'],
            'detection': detection_result,
            'session_id': request.sid
        }

        # Sauvegarde résultat en base
        if current_session['serial_number']:
            db_manager.save_detection_result(current_session['serial_number'], detection_result)

        # Envoi résultat à AR
        emit('detection_result', result)

        logger.info(f"Detection completed for {current_session['serial_number']}: {detection_result['status']}")

    except Exception as e:
        logger.error(f"Error during frame capture detection: {e}")
        emit('error', {'message': f'Detection error: {str(e)}'})

@socketio.on('voice_command')
def handle_voice_command(data):
    """Traitement commandes vocales"""
    try:
        command = data.get('command', '').lower()

        if command in ['capture', 'detect', 'analyser', 'scan']:
            # Déclencher capture + détection
            handle_capture_frame(data)

        elif command in ['reset', 'nouveau', 'restart']:
            # Reset session
            global current_session
            current_session = {
                'equipment_id': None,
                'serial_number': None,
                'detection_active': False,
                'last_frame': None
            }
            emit('session_reset', {'status': 'reset'})
            logger.info("Session reset by voice command")

        elif command in ['info', 'details', 'informations']:
            # Informations équipement courant
            if current_session['serial_number']:
                equipment = db_manager.get_equipment_details(current_session['serial_number'])
                emit('equipment_info', {'equipment': equipment})
            else:
                emit('error', {'message': 'No equipment identified'})

        else:
            emit('error', {'message': f'Unknown voice command: {command}'})

    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        emit('error', {'message': f'Voice command error: {str(e)}'})

@socketio.on('manual_detection')
def handle_manual_detection(data):
    """Détection manuelle déclenchée depuis AR"""
    try:
        # Récupération frame depuis data ou utilisation de la dernière
        if 'frame' in data:
            frame_data = base64.b64decode(data['frame'])
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            frame = current_session['last_frame']

        if frame is None:
            emit('error', {'message': 'No frame available for detection'})
            return

        # Détection complète
        detection_result = detection_manager.full_detection_pipeline(frame)

        # Envoi résultat
        emit('detection_complete', {
            'timestamp': datetime.now().isoformat(),
            'result': detection_result,
            'equipment_id': current_session['equipment_id']
        })

        logger.info(f"Manual detection completed: {detection_result['status']}")

    except Exception as e:
        logger.error(f"Error in manual detection: {e}")
        emit('error', {'message': f'Manual detection error: {str(e)}'})

# ==== GESTION DES ERREURS ====

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

# ==== DÉMARRAGE SERVEUR ====

def initialize_server():
    """Initialisation du serveur"""
    logger.info("Initializing AR Assembly Detection Server...")

    # Test connexion DB
    if not db_manager.test_connection():
        logger.error("Database connection failed!")
        return False

    # Test modèles IA
    if not detection_manager.test_models():
        logger.error("AI models loading failed!")
        return False

    logger.info("Server initialization completed successfully")
    return True

if __name__ == '__main__':
    if initialize_server():
        logger.info("Starting AR Assembly Detection Server...")
        logger.info("Server endpoints:")
        logger.info("  - Main server: http://0.0.0.0:5000")
        logger.info("  - WebSocket: ws://0.0.0.0:5000")
        logger.info("  - Sensor data: POST /sensors/<serial_number>")
        logger.info("  - Equipment info: GET /equipment/<serial_number>")
        logger.info("  - Health check: GET /health")

        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            log_output=True
        )
    else:
        logger.error("Server initialization failed. Exiting...")
        exit(1)