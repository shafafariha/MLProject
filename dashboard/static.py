import os
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import joblib
import warnings

# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand landmarks
def extract_landmarks(image, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    ) as hands:
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

# Streamlit app main function
def main():
    st.title('🖼️ BISINDO Classification (Static Image)')

    st.warning("To use the real-time webcam version, please run `dashboard.py` locally.")

    # Sidebar - model selection (disabled for now)
    st.sidebar.header('Model Selection')
    model = st.sidebar.selectbox('Choose Model', ['RF_BISINDO_99'], disabled=True)

    # Sidebar - model parameters
    st.sidebar.header('Model Settings')
    min_detection_confidence = st.sidebar.slider(
        'Min Detection Confidence', 0.0, 1.0, 0.5)
    min_tracking_confidence = st.sidebar.slider(
        'Min Tracking Confidence', 0.0, 1.0, 0.5)

    # Load model
    model_path = f'model/{model.lower()}.pkl'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return

    # Sidebar - image input
    st.sidebar.header('Image Input')
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    predefined_images = os.listdir('images/example')
    selected_image = st.sidebar.selectbox('Or choose from sample images', predefined_images)

    # Load and process image
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
    else:
        image_path = os.path.join('images/example', selected_image)
        image = cv2.imread(image_path)

    prediction_header = st.empty()

    if image is not None:
        landmarks, hand_landmarks = extract_landmarks(
            image,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        if landmarks:
            landmarks_np = np.array(landmarks).reshape(1, -1)
            prediction = clf.predict(landmarks_np)[0]

            # Draw landmarks
            for hand_landmark in hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Show results
            prediction_header.header(f"Predicted Sign: {prediction}")
            st.image(image, channels="BGR", caption=f"Predicted Sign: {prediction}")
        else:
            st.warning("⚠️ No hand landmarks detected. Try another image.")

    st.markdown("---")

if __name__ == "__main__":
    main()
