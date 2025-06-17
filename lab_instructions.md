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


```bash
python run_patchcore.py --dataset_path ./dataset/patchcore/ --category screw \
  --method patchcore --save_segmentation_images --save_plots --device cpu
```









```bash
```