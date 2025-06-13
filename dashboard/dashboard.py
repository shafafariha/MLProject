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
    """Initialize camera with better error handling for local and cloud environments"""
    try:
        if is_cloud_environment():
            st.warning(
                "⚠️ Running in cloud environment. Camera access might be limited directly.")

        # Try common camera indices
        for index in [0, 1]:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                return cap
            cap.release() # Release if not opened

        # If no camera found after trying indices
        st.error("""
        🎥 Camera not available. 
        
        If you're running this on Streamlit Cloud:
        - Direct camera access via `cv2.VideoCapture` is limited. Consider using "Image Upload" below, or run locally.
        
        If you're running locally:
        - Make sure your camera is connected and not in use by another application.
        - Try granting camera permissions to your browser.
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

    # Initialize session state for camera
    if 'camera' not in st.session_state:
        st.session_state.camera = None
        st.session_state.camera_initialized = False

    if is_cloud_environment():
        st.warning("""
        ⚠️ Note: You're running this app in Streamlit Cloud.
        Some features like direct camera access might be limited.
        For full functionality, consider running the app locally.
        """)

    # Model selection (sidebar)
    st.sidebar.header('Choose a Model (My Future Work)')
    selected_model_name = st.sidebar.selectbox(
        'Select Model', ['RF_BISINDO_99'], disabled=True) 

    # Correctly construct the path to the model file
    current_script_dir = os.path.dirname(__file__)
    model_folder_path = os.path.join(current_script_dir, '..', 'model')
    model_filename = f"{selected_model_name.lower()}.pkl"
    model_path = os.path.join(model_folder_path, model_filename)

    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}. Please ensure '{model_filename}' is in the root 'model/' folder.")
            return
        clf = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return

    st.sidebar.header('Webcam Feed Settings')
    brightness = st.sidebar.slider('Brightness', -100, 100, 0)
    contrast = st.sidebar.slider('Contrast', -100, 100, 0)
    saturation = st.sidebar.slider('Saturation', -100, 100, 0)

    st.sidebar.header('Model Settings')
    min_detection_confidence = st.sidebar.slider(
        'Min Detection Confidence', 0.0, 1.0, 0.5)
    min_tracking_confidence = st.sidebar.slider(
        'Min Tracking Confidence', 0.0, 1.0, 0.5)

    # Main content area for camera or image
    st.subheader("Webcam Live Feed")
    run_webcam = st.checkbox('Start Webcam Live Feed')
    webcam_frame_placeholder = st.empty() # Placeholder for webcam feed
    webcam_prediction_text = st.empty()

    st.markdown("---") # Separator
    
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    uploaded_image_placeholder = st.empty() # Placeholder for uploaded image
    uploaded_image_prediction_text = st.empty()

    # Logic to handle webcam vs. image upload
    if run_webcam:
        # If webcam is selected, ensure it's initialized and run
        if not st.session_state.camera_initialized:
            st.session_state.camera = initialize_camera()
            st.session_state.camera_initialized = True

        if st.session_state.camera is not None:
            # Hide image upload elements if webcam is active
            uploaded_image_placeholder.empty()
            uploaded_image_prediction_text.empty()

            while run_webcam: # Loop continues as long as checkbox is checked
                # Re-check checkbox state inside loop to allow stopping
                run_webcam = st.checkbox('Stop Webcam Live Feed', value=True) 

                try:
                    ret, frame = st.session_state.camera.read()
                    if not ret:
                        st.error("Failed to get frame from camera. Please restart the app or try again.")
                        break # Exit loop if frame cannot be read

                    frame = cv2.flip(frame, 1) # Flip horizontally for selfie view
                    # Apply brightness and contrast
                    frame = cv2.convertScaleAbs(frame, alpha=1 + contrast/100, beta=brightness)
                    # Apply saturation (convert to HSV, adjust S, convert back to BGR)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[..., 1] = cv2.add(hsv[..., 1], saturation)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    # Extract landmarks and predict
                    landmarks, hand_landmarks = extract_landmarks(
                        frame,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )

                    predicted_label = "No Hand Detected"
                    if landmarks:
                        landmarks_np = np.array(landmarks).reshape(1, -1)
                        prediction = clf.predict(landmarks_np)
                        predicted_label = prediction[0]

                        # Draw landmarks on the frame
                        for hand_landmark in hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmark,
                                                        mp_hands.HAND_CONNECTIONS)
                    
                    # Put prediction text on frame
                    cv2.putText(frame, f'Sign: {predicted_label}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # Added LINE_AA for smoother text

                    # Display frame and prediction text
                    webcam_frame_placeholder.image(frame, channels='BGR', use_column_width=True)
                    webcam_prediction_text.text(f"Predicted Sign: {predicted_label}")
                
                except Exception as e:
                    st.error(f"Error processing webcam frame: {str(e)}")
                    break # Break the loop on error
        else:
            # If camera initialization failed, but checkbox was checked, uncheck it
            if run_webcam:
                st.checkbox('Start Webcam Live Feed', value=False, key="webcam_fallback_uncheck")
    
    elif uploaded_file is not None:
        # If a file is uploaded, process it
        # Release camera if it was active
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None
            st.session_state.camera_initialized = False
        
        # Clear webcam placeholders
        webcam_frame_placeholder.empty()
        webcam_prediction_text.empty()

        try:
            # Read image as OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1) # 1 for color image

            # Apply brightness and contrast (optional for static image, but for consistency)
            processed_image = cv2.convertScaleAbs(image, alpha=1 + contrast/100, beta=brightness)
            # Apply saturation
            hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = cv2.add(hsv[..., 1], saturation)
            processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Extract landmarks and predict
            landmarks, hand_landmarks = extract_landmarks(
                processed_image,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

            predicted_label = "No Hand Detected"
            if landmarks:
                landmarks_np = np.array(landmarks).reshape(1, -1)
                prediction = clf.predict(landmarks_np)
                predicted_label = prediction[0]

                # Draw landmarks on the image
                for hand_landmark in hand_landmarks:
                    mp_drawing.draw_landmarks(processed_image, hand_landmark,
                                                mp_hands.HAND_CONNECTIONS)
            
            # Put prediction text on image
            cv2.putText(processed_image, f'Sign: {predicted_label}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display processed image and prediction text
            uploaded_image_placeholder.image(processed_image, channels='BGR', use_column_width=True)
            uploaded_image_prediction_text.text(f"Predicted Sign: {predicted_label}")

        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")
            # Clear placeholders on error
            uploaded_image_placeholder.empty()
            uploaded_image_prediction_text.empty()
            
    else: # Neither webcam nor image uploaded/active
        # Ensure camera is released if no longer needed
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None
            st.session_state.camera_initialized = False
        
        # Clear all placeholders when nothing is active
        webcam_frame_placeholder.empty()
        webcam_prediction_text.empty()
        uploaded_image_placeholder.empty()
        uploaded_image_prediction_text.empty()


    # Project Information section
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.markdown("### 👨‍💻 Project Information")
            # Removed st.info block as per request
        st.markdown("---")


if __name__ == "__main__":
    main()
