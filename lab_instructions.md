## Déploiement et Utilisation de l’AR Web/Backend (Juin 2025)

### 1. Prérequis

- **Conda** (miniconda recommandé)
- Python 3.11
- Accès à un shell (CMD/PowerShell/bash)
- Pour la commande vocale : micro fonctionnel

---

### 2. Environnement conda conseillé

```bash
conda create -n ai_env python=3.11
conda activate ai_env
conda install -y -c defaults numpy flask requests PyYAML
conda install -y -c conda-forge pyzbar Pillow qrcode opencv SpeechRecognition psycopg2 scikit-learn albumentations matplotlib seaborn flask-socketio
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch
conda install -y flask-sqlalchemy python-socketio websocket-client
conda install -y -c conda-forge tensorflow ultralytics
pip install qrcode
conda install --file requirements.txt -c conda-forge -y
````

```bash
conda init
conda create -n ai_env
```

```bash
conda install -c conda-forge opencv
```

```bash
conda create -n ai_env python=3.11
conda activate ai_env
conda install -c conda-forge opencv
pip install -r requirements.txt 
pip install qrcode
```

```bash
set FLASK_APP=test.py
set FLASK_ENV=development
set FLASK_DEBUG=0
C:\Users\yoann\miniconda3\envs\ai_env\python.exe -m flask run
```
```bash
 C:\Users\yoann\miniconda3\envs\ai_env\python.exe C:\Users\yoann\Documents\School\X4\Recherche\Human-Centered-AR-Defect-Detection\test.py
```
On another terminal
```bash
 C:\Users\yoann\miniconda3\envs\ai_env\python.exe C:\Users\yoann\Documents\School\X4\Recherche\Human-Centered-AR-Defect-Detection\client.py
```

```bash
conda install -c conda-forge psycopg2
pip install ultralytics
```

```bash
conda install scikit-learn
```
```bash
python -m reset_dataset_from_clean_copy
# Augmentation
#python -m augment_with_mask_sync #plus utile
```
```bash
#python -m generate_empty_labels_for_good
#python -m convert_masks_to_yolo
#python -m prepare_yolo_dataset
#python -m generate_dataset_yaml

```
```bash
#python -m build_classification_dataset
#python -m train_classifier
#python -m predict_classifier
#python -m run_cascade_pipeline

```

[//]: # (```bash)

[//]: # (yolo detect train data=dataset/screw/dataset.yaml model=models/yolov8n.pt epochs=20 batch=2 imgsz=320 device=cpu workers=0 cache=False)

[//]: # (move runs\detect\train4\weights\best.pt models\best_screw.pt)

[//]: # (yolo detect predict model=models\best_screw.pt source=<path_to_test_images>)

[//]: # (```)

[//]: # ()
[//]: # (```bash)

[//]: # ( yolo detect train cfg=train_config.yaml)

[//]: # (```)

[//]: # (```bash)

[//]: # ( yolo detect train cfg=train_config_v2.yaml)

[//]: # (```)

For the problem with OpenMP where there is a duplicate dll file go to the python module and and add a prefix ".DI"
Usally located at ```C:\Users\%USER%\miniconda3\envs\ai_env\Library\bin```[1] and ```C:\Users\%USER%\miniconda3\Library\bin```[2].
You should rename the [1].

```bash
conda activate ai_env

# Install core scientific packages from defaults and conda-forge
conda install -y -c defaults numpy flask requests PyYAML
conda install -y -c conda-forge pyzbar Pillow qrcode opencv SpeechRecognition psycopg2 scikit-learn albumentations matplotlib seaborn flask-socketio

# Install PyTorch and Torchvision specifically for CPU
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch

# Install Flask-SQLAlchemy and python-socketio (likely from defaults)
conda install -y flask-sqlalchemy python-socketio websocket-client

# Install Ultralytics and Tensorflow, trying conda-forge first for TensorFlow
conda install -y -c conda-forge tensorflow
conda install -y -c conda-forge ultralytics
```


[//]: # (```bash)

[//]: # (python run_patchcore.py --dataset_path ./dataset/patchcore/ --category screw \)

[//]: # (  --method patchcore --save_segmentation_images --save_plots --device cpu)

[//]: # (```)



```bash
python -m server
python -m client
```
## 3. Simuler un ESP32 : Génération & Push Automatique de Données Capteurs
### Présentation Pour tester l’écosystème AR/IoT même sans hardware réel, un script Python permet de simuler un ESP32 qui : 
- génère toutes les 1-2 secondes des données capteurs (température, vibration) 
- expose localement une route HTTP pour visualiser la dernière valeur simulée 
- envoie chaque valeur automatiquement à l’API centrale (route `/sensors/<serial_number>`) via HTTP POST, exactement comme le ferait un vrai module embarqué --- 
### Installation 
Aucun module additionnel nécessaire (tout est inclus dans les prérequis du projet). 
Vérifie juste que Flask et Requests sont bien installés dans ton environnement. --- 
### Lancer le simulateur Dans un terminal activé sur l’environnement de ton choix :
```bash
 python esp32_sensor_sim.py -id=SN-IM2025001 -api=http://192.168.1.50:5000 -port=8081 
 ``` 
- `-id` (**obligatoire**) : le numéro de série/ID simulé (doit correspondre à un équipement déclaré côté DB si tu veux la remontée complète) 
- `-api` (**optionnel**) : URL de l’API AR principale (défaut : `http://127.0.0.1:5000`) 
- `-port` (**optionnel**) : port local HTTP utilisé pour la diffusion locale (défaut : 8081) 
**Exemples** ```bash python esp32_sensor_sim.py -id=SN-IM2025007 python esp32_sensor_sim.py -id=SN-IM2025012 -api=http://192.168.1.32:5000 -port=8091 ``` --- 
### Fonctionnement - Le script démarre un mini-serveur Flask local : 
Accès possible à `http://localhost:8081/SN-IM2025001` pour la dernière valeur simulée à tout moment. 
- Toutes les 1 à 2 secondes, une valeur aléatoire de température et vibration (ex : `{"temperature": 26.41, "vibration": 1.247, ...}`) est : 
- - sauvegardée localement (pour la route HTTP) - envoyée à l’API via un POST `http://<API>/sensors/<serial_number>` (structure conforme à la spec du serveur) 
- En cas d’échec réseau, le script réessaie automatiquement et affiche l’erreur. --- 
### Multi-simulation / tests 
- **Simuler plusieurs ESP32** : lance plusieurs instances du script, chacune avec un ID différent et un port HTTP différent 
- **Debug/monitoring**: Tu peux faire un `curl` sur la route locale ou observer la console pour vérifier que les envois se passent bien et que les retours API sont OK --- 
### Exemple de log console 
``` bash
[ESP32_SIM] Serveur local sur http://0.0.0.0:8081/SN-IM2025001 
[ESP32_SIM] Envoi périodique vers http://192.168.1.50:5000/sensors/SN-IM2025001 
[ESP32_SIM] Sent: {'temperature': 23.97, 'vibration': 1.016, 'timestamp': '2025-06-18T11:27:24.133418'} | Réponse API: success 
```  
### Pour aller plus loin - Peut être adapté pour supporter MQTT ou WebSocket si besoin de tests plus avancés 
- Pour tests de montée en charge, tu peux lancer 10/20 instances en changeant juste l’ID et le port --- 
- **Ce module permet de valider tout le flux IoT/AR côté logiciel et d’alimenter dashboards, logs et tests côté client web sans attendre la livraison du hardware réel.**






```bash
```