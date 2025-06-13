import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import platform
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def is_cloud_environment():
    """Check if running in Streamlit Cloud"""
    return os.getenv('STREAMLIT_RUNTIME_ENVIRONMENT') == 'cloud'


def extract_landmarks(image, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """
    Extracts hand landmarks from an image using MediaPipe.
    Returns normalized landmarks and MediaPipe's raw hand_landmarks.
    """
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                            min_detection_confidence=min_detection_confidence,
                            min_tracking_confidence=min_tracking_confidence) as hands:
        # Convert BGR image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            # If only one hand is detected, pad with zeros for the second hand's 21 landmarks * 3 coordinates
            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0] * 63) 
            return landmarks, results.multi_hand_landmarks
    return None, None


def initialize_camera():
    """Initialize camera with better error handling"""
    try:
        if is_cloud_environment():
            st.warning("⚠️ Running in cloud environment. Camera access might be limited.")

        for index in [0, 1]:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                return cap
            cap.release()

        st.error("""
        🎥 Camera not available. 

        If you're running this on Streamlit Cloud:
        - Camera access is limited in cloud environments
        - For full functionality, please run the app locally

        If you're running locally:
        - Make sure your camera is connected and not in use by another application
        - Try granting camera permissions to your browser
        """)
        return None

    except Exception as e:
        st.error(f"""
        ❌ Error initializing camera: {str(e)}

        System Info:
        - OS: {platform.system()}
        - Python: {platform.python_version()}
        - OpenCV: {cv2.__version__}
        """)
        return None


def main():
    st.title('Real-Time BISINDO Classification')

    if 'camera' not in st.session_state:
        st.session_state.camera = None
        st.session_state.camera_initialized = False

    if is_cloud_environment():
        st.warning("⚠️ You're running this app in Streamlit Cloud. Some features like camera access might be limited.")

    # 🔍 Auto-load all models in model folder
    # CORRECTED PATH: Go up one directory from dashboard.py, then into 'model'
    current_script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(current_script_dir, '..', 'model')
    model_dir = os.path.abspath(model_dir) # Convert to absolute path for robustness

    available_models = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

    if not available_models:
        st.error("❌ No model files found in the `model/` folder.")
        return

    st.sidebar.header('Choose a Model')
    selected_model = st.sidebar.selectbox('Select Model', available_models)
    model_path = os.path.join(model_dir, selected_model)

    st.sidebar.header('Webcam Feed Settings')
    brightness = st.sidebar.slider('Brightness', -100, 100, 0)
    contrast = st.sidebar.slider('Contrast', -100, 100, 0)
    saturation = st.sidebar.slider('Saturation', -100, 100, 0)

    st.sidebar.header('Model Settings')
    min_detection_confidence = st.sidebar.slider('Min Detection Confidence', 0.0, 1.0, 0.5)
    min_tracking_confidence = st.sidebar.slider('Min Tracking Confidence', 0.0, 1.0, 0.5)

    try:
        clf = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return

    # --- START OF IMAGE UPLOAD SECTION ---
    st.header("Image Upload")
    st.write("Upload an image for classification (alternative to webcam)")

    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False, 
        help="Limit 200MB per file • JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        # Convert uploaded file to OpenCV image format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, 1) # 1 for color image

        # Process the uploaded image
        st.subheader("Classification Result from Uploaded Image:")
        
        landmarks_img, hand_landmarks_img = extract_landmarks(
            uploaded_image,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        if landmarks_img:
            landmarks_np_img = np.array(landmarks_img).reshape(1, -1)
            prediction_img = clf.predict(landmarks_np_img)
            predicted_label_img = prediction_img[0]

            # Draw landmarks on the uploaded image
            for hand_landmark in hand_landmarks_img:
                mp_drawing.draw_landmarks(uploaded_image, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Display the result on the image
            cv2.putText(uploaded_image, f'Sign: {predicted_label_img}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            st.image(uploaded_image, caption=f"Uploaded Image with Prediction: {predicted_label_img}", channels='BGR', use_column_width=True)
            st.write(f"**Predicted Sign (from image):** {predicted_label_img}")
        else:
            st.warning("No hands detected in the uploaded image.")

    # --- END OF IMAGE UPLOAD SECTION ---


    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    prediction_text = st.empty()

    if run:
        if not st.session_state.camera_initialized:
            st.session_state.camera = initialize_camera()
            st.session_state.camera_initialized = True

        if st.session_state.camera is not None:
            while run:
                try:
                    ret, frame = st.session_state.camera.read()
                    if not ret:
                        st.error("Failed to get frame from camera")
                        break

                    frame = cv2.flip(frame, 1)
                    frame = cv2.convertScaleAbs(frame, alpha=1 + contrast / 100, beta=brightness)

                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[..., 1] = cv2.add(hsv[..., 1], saturation)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    landmarks, hand_landmarks = extract_landmarks(
                        frame,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )

                    if landmarks:
                        landmarks_np = np.array(landmarks).reshape(1, -1)
                        prediction = clf.predict(landmarks_np)
                        predicted_label = prediction[0]

                        cv2.putText(frame, f'Sign: {predicted_label}', (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        for hand_landmark in hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

                        prediction_text.text(f"Predicted Sign: {predicted_label}")

                    FRAME_WINDOW.image(frame, channels='BGR')
                except Exception as e:
                    st.error(f"Error processing frame: {str(e)}")
                    break
    else:
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None
            st.session_state.camera_initialized = False


if __name__ == "__main__":
    main()
