import os
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import joblib
import warnings

# Suppress warnings and unnecessary logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function: Extract landmarks from a frame
def extract_landmarks(image, min_detection_conf=0.5, min_tracking_conf=0.5):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=min_detection_conf,
                        min_tracking_confidence=min_tracking_conf) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0] * 63)  # Pad for second hand
            return landmarks, results.multi_hand_landmarks
    return None, None

# Function: Initialize webcam
def initialize_camera():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open camera.")
            return None
        return cap
    except Exception as e:
        st.error(f"❌ Error initializing camera: {str(e)}")
        return None

# Main Streamlit App
def main():
    st.title('🖐️ Real-Time BISINDO Classification')

    # Setup camera state
    if 'camera' not in st.session_state:
        st.session_state.camera = None

    # Sidebar - Model and webcam settings
    st.sidebar.header('Model & Webcam Settings')
    model_name = st.sidebar.selectbox('Select Model', ['RF_BISINDO_99'], disabled=True)
    brightness = st.sidebar.slider('Brightness', -100, 100, 0)
    contrast = st.sidebar.slider('Contrast', -100, 100, 0)
    saturation = st.sidebar.slider('Saturation', -100, 100, 0)
    min_detection_conf = st.sidebar.slider('Min Detection Confidence', 0.0, 1.0, 0.5)
    min_tracking_conf = st.sidebar.slider('Min Tracking Confidence', 0.0, 1.0, 0.5)

    # Load model
    try:
        model_path = f'model/{model_name.lower()}.pkl'
        clf = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return

    # UI elements
    run = st.checkbox('▶️ Start Webcam')
    FRAME_WINDOW = st.image([])
    prediction_text = st.empty()

    # Webcam processing loop
    if run:
        if st.session_state.camera is None:
            st.session_state.camera = initialize_camera()

        if st.session_state.camera is not None:
            while run:
                ret, frame = st.session_state.camera.read()
                if not ret:
                    st.error("❌ Failed to read frame from camera.")
                    break

                # Apply adjustments
                frame = cv2.flip(frame, 1)
                frame = cv2.convertScaleAbs(frame, alpha=1 + contrast / 100, beta=brightness)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[..., 1] = cv2.add(hsv[..., 1], saturation)
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # Hand landmarks & prediction
                landmarks, hand_landmarks = extract_landmarks(
                    frame, min_detection_conf, min_tracking_conf)

                if landmarks:
                    landmarks_np = np.array(landmarks).reshape(1, -1)
                    predicted_label = clf.predict(landmarks_np)[0]

                    # Draw results
                    cv2.putText(frame, f'Sign: {predicted_label}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    for hand_landmark in hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
                    prediction_text.text(f"Predicted Sign: {predicted_label}")

                FRAME_WINDOW.image(frame, channels='BGR')
    else:
        # Release camera when stopped
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None

    # Footer line
    st.markdown("---")


if __name__ == "__main__":
    main()
