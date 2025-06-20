from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

def estimate_hr(ppg_signal, fps):
    n = len(ppg_signal)
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    fft = np.abs(np.fft.rfft(ppg_signal - np.mean(ppg_signal)))
    idx = np.argmax(fft)
    hr = freqs[idx] * 60
    return hr

@app.route('/vitals', methods=['POST'])
def vitals():
    if 'frame' not in request.files:
        return jsonify({'error':'No frame'}), 400
    file = request.files['frame']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return jsonify({'error':'No face detected'}), 200
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
    y_coords = [y for (x,y) in pts]
    ppg = y_coords[:100]
    hr = estimate_hr(ppg, fps=30)
    bp = 0.5 * hr + 40
    hydration = max(0, min(100, 100 - (hr-60)))
    return jsonify({'heart_rate': round(hr,2), 'blood_pressure': round(bp,2), 'hydration_percent': round(hydration,2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
