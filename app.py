from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from collections import Counter
import pandas as pd
import os
import time

app = Flask(__name__)

# Load the emotion detection model
def create_emotion_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    return model

emotion_model = create_emotion_model()
emotion_model.load_weights("emotion_model.h5")
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize webcam
cap = cv2.VideoCapture(0)
emotion_counts = []

def generate_frames():
    global emotion_counts
    start_time = time.time()
    checkpoint_duration = 10  # seconds

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            detected_emotion = emotion_dict[maxindex]
            emotion_counts.append(detected_emotion)
            cv2.putText(frame, detected_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Checkpoint for emotion detection
        if time.time() - start_time > checkpoint_duration:
            if emotion_counts:
                final_emotion = Counter(emotion_counts).most_common(1)[0][0]
            else:
                final_emotion = "Neutral"
            emotion_counts = []
            start_time = time.time()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if emotion_counts:
        final_emotion = Counter(emotion_counts).most_common(1)[0][0]
    else:
        final_emotion = "Neutral"
    recommended_songs = recommend_songs(final_emotion)
    return jsonify({
        'emotion': final_emotion,
        'songs': recommended_songs
    })

def recommend_songs(emotion):
    emotion_to_csv = {
        "Angry": "angry.csv",
        "Disgusted": "disgusted.csv",
        "Fearful": "fearful.csv",
        "Happy": "happy.csv",
        "Neutral": "neutral.csv",
        "Sad": "sad.csv",
        "Surprised": "surprised.csv"
    }
    csv_file = f'songs/{emotion_to_csv.get(emotion, "neutral.csv")}'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df.to_dict(orient='records')
    else:
        return []

if __name__ == '__main__':
    app.run(debug=True)
