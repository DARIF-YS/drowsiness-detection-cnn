import gradio as gr 
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import winsound
import os

# Configuration
IMG_SIZE        = 64
ALERT_THRESHOLD = 3  
MODEL_PATH      = "dl-model/eye_state_model.keras"

# Charger le modèle si disponible
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Modèle chargé.")
else:
    print(f"ERREUR : fichier {MODEL_PATH} introuvable.")
    model = None  # Modèle vide pour tests sans fichier (à enlever en prod)

# Charger les cascades pour détecter les yeux
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")

# Variable globale pour suivre le temps où les yeux sont fermés
closed_start_time = None

def predict_eye(gray_eye):
    # Prédit si l'œil est ouvert ou fermé
    if model is None:
        return "Open"  # Si pas de modèle, on suppose ouvert
    
    eye = cv2.resize(gray_eye, (IMG_SIZE, IMG_SIZE)) / 255.0
    eye = np.expand_dims(eye, axis=(0, -1))
    pred = model.predict(eye, verbose=0)[0][0]
    return "Closed" if pred < 0.5 else "Open"

def detect_and_draw(cascade, gray, frame):
    # Détecte un œil et dessine un rectangle autour
    eyes = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30, 30))
    
    if len(eyes) == 0:
        return 1  # Pas d'œil détecté = considéré fermé pour la sécurité
    
    x, y, w, h = eyes[0]
    roi_gray = gray[y:y+h, x:x+w]
    state = predict_eye(roi_gray)
    
    # Couleur verte si ouvert, rouge si fermé (en RGB)
    color = (0, 255, 0) if state == "Open" else (255, 0, 0)
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, state, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return 1 if state == "Closed" else 0

def process_frame(input_image):
    # Fonction principale appelée par Gradio pour chaque image
    global closed_start_time

    if input_image is None:
        return None

    frame = np.array(input_image)  # Convertit en tableau numpy
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convertit en niveaux de gris

    # Compte le nombre d'yeux fermés détectés
    eyes_closed_count = detect_and_draw(left_eye_cascade, gray, frame) + \
                        detect_and_draw(right_eye_cascade, gray, frame)

    if eyes_closed_count >= 2:
        if closed_start_time is None:
            closed_start_time = time.time()
        
        elapsed = time.time() - closed_start_time
        progress = min(elapsed / ALERT_THRESHOLD, 1)
        
        # Affiche une barre de progression rouge
        cv2.rectangle(frame, (50, 50), (350, 80), (255, 255, 255), 2)
        cv2.rectangle(frame, (50, 50), (50 + int(progress * 300), 80), (255, 0, 0), -1)
        
        text = f"Eyes closed / Not detected : {elapsed:.1f}s"
        cv2.putText(frame, text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if elapsed >= ALERT_THRESHOLD:
            cv2.putText(frame, "DROWSINESS ALERT!", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            # Fait un beep (Windows uniquement)
            try:
                winsound.Beep(1000, 200)
            except:
                pass  # Ignore les erreurs sur autres OS
    else:
        closed_start_time = None
        cv2.rectangle(frame, (50, 50), (350, 80), (255, 255, 255), 2)
        cv2.putText(frame, "Eyes open", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Interface Gradio
with gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True),  # Utilise la webcam en streaming
    outputs="image",
    live=True,
    flagging_mode="manual",
    title="Détection de Somnolence en Temps Réel",
    description="Regardez la caméra. Si vos yeux restent fermés ou non détectés plus de 3 secondes, une alerte se déclenche."
) as demo:
    demo.launch()