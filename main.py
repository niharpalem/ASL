import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from scipy.spatial.distance import euclidean
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Load the saved models and preprocessing objects
@st.cache_resource
def load_models():
    svm_model = joblib.load('svm_model.joblib')
    lgbm_model = joblib.load('lgbm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    pca = joblib.load('pca.joblib')
    return svm_model, lgbm_model, scaler, pca

svm_model, lgbm_model, scaler, pca = load_models()

# Initialize and cache MediaPipe Hands
@st.cache_resource
def load_mediapipe_model():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

hands = load_mediapipe_model()
mp_drawing = mp.solutions.drawing_utils

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

# Create placeholders for the results
results_placeholder = st.empty()

# Upload image using Streamlit's file uploader
uploaded_file = st.file_uploader("Upload an image of an ASL sign", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image_bytes = uploaded_file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(image_rgb)

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

        # Get predictions from both models
        svm_prediction = svm_model.predict(features_pca)
        svm_probabilities = svm_model.predict_proba(features_pca)[0]
        lgbm_prediction = lgbm_model.predict(features_pca)
        lgbm_probabilities = lgbm_model.predict_proba(features_pca)[0]

        # Concatenate results
        combined_probabilities = (svm_probabilities + lgbm_probabilities) / 2
        combined_prediction = np.argmax(combined_probabilities)

        # Draw hand landmarks on the image
        mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Create a figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the captured frame
        ax1.imshow(image_rgb)
        ax1.set_title('Uploaded Image')
        ax1.axis('off')

        # Plot the extracted points in 2D space
        ax2.scatter(landmarks[:, 0], landmarks[:, 1])
        ax2.set_title('Extracted Points')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.invert_yaxis()

        # Plot the word probabilities
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        ax3.bar(alphabet, combined_probabilities)
        ax3.set_title('Combined Word Probabilities')
        ax3.set_xlabel('Letters')
        ax3.set_ylabel('Probability')
        ax3.tick_params(axis='x', rotation=90)

        plt.tight_layout()

        # Display the results
        results_placeholder.pyplot(fig)

        # Display the predicted letter
        st.write(f"Predicted ASL Letter: {alphabet[combined_prediction]}")
        st.write(f"SVM Prediction: {svm_prediction[0]}")
        st.write(f"LightGBM Prediction: {alphabet[lgbm_prediction[0]]}")
    else:
        st.write("No hand detected in the image.")
