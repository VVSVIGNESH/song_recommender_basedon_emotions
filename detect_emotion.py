import cv2
import numpy as np
import tensorflow as tf
from collections import Counter

# Define the model architecture
def create_emotion_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(48, 48, 1), name="conv2d_input"),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv2d"),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"),
        tf.keras.layers.Dropout(0.25, name="dropout"),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name="conv2d_3"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_2"),
        tf.keras.layers.Dropout(0.25, name="dropout_1"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(1024, activation='relu', name="dense"),
        tf.keras.layers.Dropout(0.5, name="dropout_2"),
        tf.keras.layers.Dense(7, activation='softmax', name="dense_1")
    ])
    return model

# Initialize the model
emotion_model = create_emotion_model()

# Load weights into the model
emotion_model.load_weights("Emotion-Based-Music-Recommendation-System-main/model/emotion_model.h5")

# Define emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize emotion counts
emotion_counts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the face detector
    face_detector = cv2.CascadeClassifier(r'C:\Users\vigne\Downloads\Emotion-Based-Music-Recommendation-System-main\Emotion-Based-Music-Recommendation-System-main\haarcascades\haarcascade_frontalface_default.xml')
    
    # Detect faces
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        detected_emotion = emotion_dict[maxindex]
        
        # Store the detected emotion
        emotion_counts.append(detected_emotion)
        cv2.putText(frame, detected_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Determine the final emotion
if emotion_counts:
    final_emotion = Counter(emotion_counts).most_common(1)[0][0]
    print(f"Final detected emotion: {final_emotion}")
else:
    final_emotion = "Neutral"
    print("No emotion detected")

# Save the result to a file
with open('detected_emotion.txt', 'w') as file:
    file.write(final_emotion)

