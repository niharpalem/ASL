import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from numpy.linalg import norm

# Load the saved Random Forest model
@st.cache_resource
def load_model():
    rf_model = joblib.load('best_random_forest_model.pkl')
    return rf_model

rf_model = load_model()

# Initialize and cache MediaPipe Hands
@st.cache_resource
def load_mediapipe_model():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

hands = load_mediapipe_model()
mp_drawing = mp.solutions.drawing_utils

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

        # Calculate angles
        angles = calculate_angles(landmarks)

        # Convert angles to a feature array
        features = np.array(angles).reshape(1, -1)

        # Log feature details
        st.write("Features shape:", features.shape)
        st.write("Features:", features)

        # Check for NaN or infinite values
        if np.isnan(features).any() or np.isinf(features).any():
            st.write("Invalid feature values detected.")
        else:
            # Get predictions from the Random Forest model
            rf_prediction = rf_model.predict(features)
            rf_probabilities = rf_model.predict_proba(features)[0]

            # Display the predicted letter
            st.write(f"Random Forest Prediction: {alphabet[rf_prediction[0]]}")
            
            # Display top 3 predictions
            top_3_indices = np.argsort(rf_probabilities)[-3:][::-1]
            st.write("Top 3 Predictions:")
            for idx in top_3_indices:
                st.write(f"{alphabet[idx]}: {rf_probabilities[idx]:.4f}")

    else:
        st.write("No hand detected in the image.")
