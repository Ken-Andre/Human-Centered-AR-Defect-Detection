import speech_recognition as sr
import requests

recognizer = sr.Recognizer()
server_ip = "192.168.1.100"
port = "8080"

while True:
    with sr.Microphone() as source:
        print("Dites une commande...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio, language="fr-FR")
            print(f"Commande : {command}")
            if "afficher données capteur" in command.lower() or "print data" in command.lower():
                try:
                    response = requests.get(f"http://{server_ip}:{port}/SN-IM2025007")
                    print(f"Données capteurs : {response.json()}")
                except:
                    print("Aucune donnée de capteurs n’est reçue, assurez-vous de rester stable en face du QR code pour la détection et ressayez")
            elif "afficher documentation" in command.lower() or "print documentation" in command.lower():
                print("Documentation pour Industrial Motor : [lien ou texte]")
            elif "vérifier l’état" in command.lower() or "verify state" in command.lower():
                response = requests.post(f"http://{server_ip}:5000/stream", files={"frame": open("current_frame.jpg", "rb")})
                if response.json()["status"] == "equipment_detected":
                    print("État : Défectueux à 75%")  # À remplacer par la prédiction YOLO
                else:
                    print("Équipement non identifié, veuillez réessayer")
        except sr.UnknownValueError:
            print("Commande non reconnue")