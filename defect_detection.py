import tensorflow as tf
import cv2
import numpy as np

# Charger un modèle pré-entraîné
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
model = tf.keras.Sequential([model, tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(2, activation='softmax')])

# Charger une image (exemple : vis)
image = cv2.imread('screw_image.jpg')
image = cv2.resize(image, (224, 224))  # Taille attendue par MobileNet
image = np.expand_dims(image, axis=0) / 255.0  # Normalisation

# Prédiction
prediction = model.predict(image)
if prediction[0][0] > 0.5:
    print("Pas de défaut")
else:
    print("Défaut détecté")