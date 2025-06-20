from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

api = Blueprint('api', __name__)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

def estimate_hr(ppg, fps=30):
    n = len(ppg)
    freqs = np.fft.rfftfreq(n, d=1/fps)
    fft = np.abs(np.fft.rfft(ppg - np.mean(ppg)))
    return freqs[np.argmax(fft)] * 60

@api.route('/', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    file = request.files['video']
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(arr, cv2.IMREAD_COLOR))
    if not cap.isOpened():
        return jsonify({'error': 'Invalid video'}), 400

    ppg = []
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while count < 100:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue
        ys = [lm.y for lm in res.multi_face_landmarks[0].landmark]
        ppg.append(np.mean(ys))
        count += 1

    cap.release()

    if len(ppg) < 10:
        return jsonify({'error': 'No valid frames'}), 200

    hr = estimate_hr(np.array(ppg), fps)
    bp = 0.5 * hr + 40
    hydration = max(0, min(100, 100 - (hr - 60)))

    return jsonify({
        'heart_rate': round(hr, 2),
        'blood_pressure': round(bp, 2),
        'hydration_percent': round(hydration, 2)
    })
