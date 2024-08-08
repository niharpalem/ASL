import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
from numpy.linalg import norm

# Function to load the Random Forest model
@st.cache_resource
def load_model():
    try:
        return joblib.load('best_random_forest_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model using the cached function
model = load_model()

# Ensure the model is loaded before proceeding
if model is None:
    st.stop()

# Initialize MediaPipe Hands
@st.cache_resource
def load_mediapipe_model():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

hands = load_mediapipe_model()
mp_drawing = mp.solutions.drawing_utils

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    center = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - center
    max_distance = np.max(np.linalg.norm(landmarks_centered, axis=1))
    if max_distance > 0:
        landmarks_normalized = landmarks_centered / max_distance
    else:
        landmarks_normalized = landmarks_centered
    return landmarks_normalized

# Function to calculate angles between landmarks
def calculate_angles(landmarks):
    angles = []
    for i in range(20):
        for j in range(i + 1, 21):
            vector = landmarks[j] - landmarks[i]
            angle_x = np.arccos(np.clip(vector[0] / norm(vector), -1.0, 1.0))
            angle_y = np.arccos(np.clip(vector[1] / norm(vector), -1.0, 1.0))
            angles.extend([angle_x, angle_y])
    return angles

# Streamlit app
st.title("ASL Recognition App")

# Upload image using Streamlit's file uploader
uploaded_file = st.file_uploader("Upload an image of an ASL sign", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hand landmarks
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Extract and normalize landmarks
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                landmarks_normalized = normalize_landmarks(landmarks)
                
                # Calculate angles using normalized landmarks
                angles = calculate_angles(landmarks_normalized)
                
                # Prepare input with feature names
                angle_columns = [f'angle_{i}' for i in range(len(angles))]
                angles_df = pd.DataFrame([angles], columns=angle_columns)
                
                # Predict the alphabet
                probabilities = model.predict_proba(angles_df)[0]
                top_indices = np.argsort(probabilities)[::-1][:5]
                top_probabilities = probabilities[top_indices]
                top_classes = model.classes_[top_indices]
                
                # Display the top 5 predictions
                st.write("Top 5 Predicted Alphabets:")
                for i in range(5):
                    st.write(f"{top_classes[i]}: {top_probabilities[i]:.2f}")
            
            # Display the image with landmarks
            st.image(image, caption="Processed Image with Landmarks", use_column_width=True)
        else:
            st.write("No hands detected. Please try another image.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
