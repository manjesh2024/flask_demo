from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random

app = Flask(__name__)
CORS(app)

@app.route('/vitals', methods=['POST'])
def get_vitals():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    filename = video.filename
    save_path = f"/tmp/{filename}"
    video.save(save_path)

    # Get file size (optional)
    size_mb = round(os.path.getsize(save_path) / (1024 * 1024), 2)

    # Dummy vitals
    heart_rate = round(random.uniform(65, 85), 2)
    blood_pressure = round(random.uniform(110, 130), 2)
    hydration = round(random.uniform(70, 100), 2)

    return jsonify({
        'filename': filename,
        'file_size_MB': size_mb,
        'heart_rate': heart_rate,
        'blood_pressure': blood_pressure,
        'hydration_percent': hydration
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
