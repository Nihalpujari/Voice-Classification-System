from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
import numpy as np
import librosa
from speaker_data import speaker_info

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model, scaler, and label encoder
clf = joblib.load('speaker_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
    median_pitch = np.nanmedian(f0) if f0 is not None else 0
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    rms = librosa.feature.rms(y=y).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    chroma = librosa.feature.chroma_stft(y=y).mean()
    return np.hstack([mfcc, spectral_centroid, median_pitch, zcr, rms, spectral_bandwidth, chroma])


def predict_speaker(file_path, threshold=0.6):
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    probabilities = clf.predict_proba(features_scaled)[0]
    max_prob = np.max(probabilities)

    if max_prob < threshold:
        return {
            'name': "Unknown Speaker",
            'image_url': url_for('static', filename='images/unknown.jpg'),
            'age': 'N/A',
            'phone': 'N/A',
            'experience': 'N/A',
            'email': 'N/A'
        }

    predicted_class_index = np.argmax(probabilities)
    label = le.inverse_transform([predicted_class_index])[0]
    name = str(label)

    image_filename = f'{name.lower()}.jpg'
    image_path = os.path.join('static/images', image_filename)
    image_url = url_for('static', filename=f'images/{image_filename}') if os.path.exists(image_path) else url_for('static', filename='images/unknown.jpg')


    info = speaker_info.get(name, {})

    return {
        'name': name,
        'image_url': image_url,
        'age': info.get('age', 'N/A'),
        'phone': info.get('phone', 'N/A'),
        'experience': info.get('experience', 'N/A'),
        'email': info.get('email', 'N/A')
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = predict_speaker(file_path)
        return render_template('result.html', **result)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
