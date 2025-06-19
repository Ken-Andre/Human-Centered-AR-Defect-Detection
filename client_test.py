from flask import Flask, render_template_string, render_template, request, redirect, url_for, session
import requests
import socketio
import threading
import base64
import time

BACKEND_URL = "http://127.0.0.1:5000"

app = Flask(__name__, static_folder="./app/static", template_folder="./app/templates")
app.secret_key = "ar_secret"

sio = socketio.Client()
ar_state = {
    "equipment": None,
    "qr_data": None,
    "last_detection": None,
    "sensor_data": [],
}

def socketio_background_connect():
    try:
        sio.connect(BACKEND_URL)
        print("[WebSocket] Connecté au backend.")
    except Exception as e:
        print("[WebSocket] Impossible de connecter au backend:", e)

# --- Events SocketIO backend ---
@sio.event
def connect():
    print("[WebSocket] Connecté.")

@sio.event
def disconnect():
    print("[WebSocket] Déconnecté.")

@sio.on("equipment_identified")
def equipment_identified(data):
    ar_state["equipment"] = data["equipment"]
    ar_state["qr_data"] = data.get("qr_data")

@sio.on("detection_result")
def detection_result(data):
    ar_state["last_detection"] = data

@sio.on("sensor_data")
def sensor_data(data):
    ar_state["sensor_data"].append(data)

@sio.on("error")
def socket_error(data):
    print("[WebSocket] Erreur:", data)

# --- Lancer SocketIO dès le démarrage du script ---
threading.Thread(target=socketio_background_connect, daemon=True).start()


@app.route("/", methods=["GET", "POST"])
def index():
    # ... (reste de ta route index, inchangé)
    # return render_template_string("""
    #     <h2>AR Web Client (socketio + REST)</h2>
    #     <form method="post" enctype="multipart/form-data">
    #         <input type="file" name="image" required>
    #         <button type="submit">Envoyer image (stream)</button>
    #     </form>
    #     <form action="/capture" method="post" style="margin-top:1em;">
    #         <button type="submit">Détection défaut (capture_frame)</button>
    #     </form>
    #     <form action="/health" method="get" style="margin-top:1em;">
    #         <button type="submit">Health Check</button>
    #     </form>
    #     <h4>Équipement courant :</h4>
    #     <pre>{{equipment}}</pre>
    #     <a href="/doc">Voir documentation</a> |
    #     <a href="/capteurs">Voir capteurs (temps réel)</a> |
    #     <a href="/voice">Commande vocale (dev)</a>
    #     <h4>Dernier résultat de détection :</h4>
    #     <pre>{{last_detection}}</pre>
    #     <h4>État du backend :</h4>
    #     <pre>{{health}}</pre>
    #     <h4 style="color:green;">{{msg}}</h4>
    # """, equipment=ar_state["equipment"], last_detection=ar_state["last_detection"], msg="", health=session.get("health"))
    return render_template("index.html")

# ... (autres routes identiques à mon message précédent)
@app.route("/capture", methods=["POST"])
def capture():
    sio.emit("capture_frame", {})
    time.sleep(1)  # Laisse le temps au serveur de répondre (améliore avec events si besoin)
    return redirect(url_for("index"))

@app.route("/doc")
def doc():
    equipment = ar_state["equipment"]
    if not equipment or not equipment.get("serial_number"):
        return "<p>Aucun équipement courant détecté.</p><a href='/'>Retour</a>"
    serial = equipment["serial_number"]
    r = requests.get(f"{BACKEND_URL}/equipment/{serial}/documentation")
    doc = r.json()
    return render_template_string("""
        <h3>Documentation équipement {{serial}}</h3>
        <pre>{{doc}}</pre>
        <a href="/">Retour</a>
    """, serial=serial, doc=doc)

@app.route("/capteurs")
def capteurs():
    data = ar_state["sensor_data"][-10:]  # Derniers 10
    return render_template_string("""
        <h3>Capteurs (temps réel)</h3>
        {% for d in data %}
            <pre>{{d}}</pre>
        {% endfor %}
        <a href="/">Retour</a>
    """, data=data)

@app.route("/health")
def health():
    r = requests.get(f"{BACKEND_URL}/health")
    session["health"] = r.json()
    return redirect(url_for("index"))

@app.route("/voice")
def voice():
    # Placeholder pour voice_control.py
    return render_template_string("""
        <h3>Commande vocale</h3>
        <p>À venir : intégration avec <code>voice_control.py</code>.</p>
        <a href="/">Retour</a>
    """)

if __name__ == "__main__":
    app.run(port=5050, debug=True)
