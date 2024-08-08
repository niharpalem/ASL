import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from numpy.linalg import norm

# Initialize MediaPipe Hands
@st.cache_resource
def load_mediapipe_model():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

hands = load_mediapipe_model()
mp_drawing = mp.solutions.drawing_utils

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
    # Read the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hand landmarks
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            
            # Calculate angles
            angles = calculate_angles(landmarks)
            
            # Display the angles
            st.write("Calculated Angles:")
            st.write(angles)
        
        # Display the image with landmarks
        st.image(image, caption="Processed Image with Landmarks", use_column_width=True)
    else:
        st.write("No hands detected. Please try another image.")
