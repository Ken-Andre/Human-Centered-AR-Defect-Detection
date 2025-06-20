#!/usr/bin/env python3
"""
Simulateur de module ESP32 pour génération et envoi de données capteurs.

Usage :
    python esp32_sensor_sim.py -id=SN-IM2025001 [-api=http://192.168.1.50:5000] [-port=8081]
"""

import argparse
import random
import time
from datetime import datetime
import threading
from flask import Flask, jsonify
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ------------- Configuration & Argument Parsing -----------------
parser = argparse.ArgumentParser()
parser.add_argument("-id", "--serial_number", required=True, help="Numéro de série/ID unique du module (ex : SN-IM2025001)")
parser.add_argument("-api", "--api_url", default="https://127.0.0.1:5000", help="Adresse du serveur principal (par défaut: http://127.0.0.1:5000)")
parser.add_argument("-port", "--port", type=int, default=8081, help="Port local d'écoute (par défaut: 8081)")
args = parser.parse_args()

SERIAL_NUMBER = args.serial_number
API_URL = args.api_url.rstrip("/")
LOCAL_PORT = args.port
LOCAL_ROUTE = f"/{SERIAL_NUMBER}"
POST_ROUTE = f"/sensors/{SERIAL_NUMBER}"

# ------------- Génération de données capteurs -----------------
def generate_sensor_data():
    # Simule des valeurs plausibles
    return {
        "temperature": round(random.uniform(21, 30), 2),
        "vibration": round(random.uniform(0.01, 2.0), 3),
        "timestamp": datetime.now().isoformat()
    }

# ------------- Serveur local Flask pour test/debug --------------
app = Flask(__name__)
latest_data = {}

@app.route(LOCAL_ROUTE, methods=["GET"])
def get_data():
    return jsonify({"serial_number": SERIAL_NUMBER, "data": latest_data})

def run_local_server():
    print(f"[ESP32_SIM] Serveur local sur https://0.0.0.0:{LOCAL_PORT}{LOCAL_ROUTE}")
    app.run(
        host="0.0.0.0",
        port=LOCAL_PORT,
        debug=False,
        use_reloader=False,
        ssl_context=('cert.pem', 'key.pem')
    )


# ------------- Thread d'envoi périodique ------------------------
def send_to_api():
    global latest_data
    session = requests.Session()
    target_url = f"{API_URL}{POST_ROUTE}"
    print(f"[ESP32_SIM] Envoi périodique vers {target_url}")
    while True:
        try:
            data = generate_sensor_data()
            latest_data = data
            # POST vers le serveur principal (API AR Assembly Detection)
            r = session.post(target_url, json=data, timeout=3, verify=False)
            if r.status_code == 200:
                print(f"[ESP32_SIM] Sent: {data} | Réponse API: {r.json().get('status','?')}")
            else:
                print(f"[ESP32_SIM] [Erreur POST] Statut: {r.status_code}, Réponse: {r.text}")
        except Exception as e:
            print(f"[ESP32_SIM] [Erreur Envoi] {e}")
        time.sleep(random.uniform(1.0, 2.0))

# ------------- Lancement (local + background API) ---------------
if __name__ == "__main__":
    # Serveur local en thread
    threading.Thread(target=run_local_server, daemon=True).start()
    # Envoi API dans la boucle principale
    send_to_api()
