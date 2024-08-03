import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from scipy.spatial.distance import euclidean
from numpy.linalg import norm

# Load the saved model and preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load('svm_model.joblib')  # Load the SVM model
    scaler = joblib.load('scaler.joblib')
    pca = joblib.load('pca.joblib')
    return model, scaler, pca

model, scaler, pca = load_model()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Function to calculate distances
def calculate_distances(landmarks):
    distances = []
    pairs = [
        (0, [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20]),
        (1, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (2, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (3, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (4, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (5, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (6, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (7, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (8, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (9, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (10, [12, 13, 14, 15, 16, 17, 18, 19, 20]),
        (11, [13, 14, 15, 16, 17, 18, 19, 20]),
        (12, [13, 14, 15, 16, 17, 18, 19, 20]),
        (13, [15, 16, 17, 18, 19, 20]),
        (14, [16, 17, 18, 19, 20]),
        (15, [17, 18, 19, 20]),
        (16, [17, 18, 19, 20]),
        (17, [19, 20]),
        (18, [20])
    ]
    for start, ends in pairs:
        for end in ends:
            distances.append(euclidean(landmarks[start], landmarks[end]))
    return distances

# Function to calculate angles
def calculate_angles(landmarks):
    angles = []
    for i in range(20):
        for j in range(i+1, 21):
            vector = landmarks[j] - landmarks[i]
            angle_x = np.arccos(np.clip(vector[0] / norm(vector), -1.0, 1.0))
            angle_y = np.arccos(np.clip(vector[1] / norm(vector), -1.0, 1.0))
            angles.extend([angle_x, angle_y])
    return angles

# Streamlit app
st.title("ASL Recognition App")

# Create a placeholder for the video feed
video_placeholder = st.empty()

# Create a placeholder for the prediction
prediction_placeholder = st.empty()

# Start the webcam
cap = cv2.VideoCapture(0)

last_process_time = time.time()
process_interval = 5  # seconds

while True:
    success, image = cap.read()
    if not success:
        st.error("Failed to capture image from camera.")
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(image_rgb)

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    if current_time - last_process_time >= process_interval:
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Assuming only one hand

            # Extract landmark coordinates
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

            # Normalize landmarks
            landmarks[:, 0] = (landmarks[:, 0] - landmarks[:, 0].min()) / (landmarks[:, 0].max() - landmarks[:, 0].min())
            landmarks[:, 1] = (landmarks[:, 1] - landmarks[:, 1].min()) / (landmarks[:, 1].max() - landmarks[:, 1].min())

            # Calculate distances and angles
            distances = calculate_distances(landmarks)
            angles = calculate_angles(landmarks)

            # Combine features
            features = np.array(distances + angles).reshape(1, -1)

            # Scale features
            features_scaled = scaler.transform(features)

            # Apply PCA
            features_pca = pca.transform(features_scaled)

            # Make prediction
            prediction = model.predict(features_pca)

            # Update prediction placeholder
            prediction_placeholder.text(f"Predicted ASL Letter: {prediction[0]}")

        last_process_time = current_time

    # Update video feed
    video_placeholder.image(image_rgb, channels="RGB")

    # Check if the user wants to stop the app
    if st.button('Stop'):
        break

# Release the webcam
cap.release()
