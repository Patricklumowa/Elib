from flask import Flask, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Mapping of model predictions to labels
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.json
    landmarks = data['landmarks']  # Expecting landmarks as input

    # Normalize and prepare data for prediction
    data_aux = []
    x_ = [landmark[0] for landmark in landmarks]
    y_ = [landmark[1] for landmark in landmarks]

    for i in range(len(landmarks)):
        data_aux.append(landmarks[i][0] - min(x_))
        data_aux.append(landmarks[i][1] - min(y_))

    # Make prediction
    try:
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
    except Exception as e:
        print(f"Prediction error: {e}")
        predicted_character = '?'

    return jsonify({'predicted_character': predicted_character})

if __name__ == '__main__':
    app.run()