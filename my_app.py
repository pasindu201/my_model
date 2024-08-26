from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('sound_classification_model.h5')

def preprocess_audio(audio_data):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=80)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_reshaped = mfcc_mean.reshape(1, 80, 1)
    return mfcc_reshaped

@app.route('/')
def home():
    return jsonify({'message': 'Hello World!'})

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    audio_data, sr = librosa.load(audio_file)
    
    # Preprocess the audio to get MFCC features
    mfcc_reshaped = preprocess_audio(audio_data)
    
    # Get the model prediction
    prediction = model.predict(mfcc_reshaped)
    predicted_class = np.argmax(prediction, axis=1)
    
    return jsonify({'prediction': int(predicted_class[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
