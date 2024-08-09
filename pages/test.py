import streamlit as st
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

class FaceMeshTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
    
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        results = self.face_mesh.process(image)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec
                )
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit app title
st.title("Live Face Mesh with MediaPipe")

webrtc_streamer(key="face-mesh", video_transformer_factory=FaceMeshTransformer)
