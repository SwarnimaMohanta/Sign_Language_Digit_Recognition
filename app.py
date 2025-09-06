import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.hands import HandLandmark
from PIL import Image
import tensorflow as tf

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("hand_digit_model_best.keras")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        try:
            return tf.keras.models.load_model("hand_digit_model_best.keras", compile=False)
        except Exception as e2:
            st.error(f"Failed to load model: {str(e2)}")
            return None

# ========================
# MEDIAPIPE SETUP
# ========================
@st.cache_resource
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return mp_hands, mp_drawing, hands

# ========================
# FINGER COUNTING FUNCTION
# ========================
def count_fingers(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    fingers = []
    
    # Thumb
    thumb_tip = lm[HandLandmark.THUMB_TIP]
    thumb_mcp = lm[HandLandmark.THUMB_MCP]
    
    if hand_label == "Right":
        thumb_extended = thumb_tip.x > thumb_mcp.x + 0.015
    else:
        thumb_extended = thumb_tip.x < thumb_mcp.x - 0.015
    
    fingers.append(1 if thumb_extended else 0)
    
    # Other fingers
    fingers.append(1 if lm[HandLandmark.INDEX_FINGER_TIP].y < lm[HandLandmark.INDEX_FINGER_PIP].y else 0)
    fingers.append(1 if lm[HandLandmark.MIDDLE_FINGER_TIP].y < lm[HandLandmark.MIDDLE_FINGER_PIP].y else 0)
    fingers.append(1 if lm[HandLandmark.RING_FINGER_TIP].y < lm[HandLandmark.RING_FINGER_PIP].y else 0)
    fingers.append(1 if lm[HandLandmark.PINKY_TIP].y < lm[HandLandmark.PINKY_PIP].y else 0)

    return fingers, sum(fingers)

# ========================
# IMAGE PROCESSING FUNCTION
# ========================
def process_image(image, model, hands):
    # Convert PIL to OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv = cv2.flip(image_cv, 1)
    h, w, c = image_cv.shape
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    total_fingers = 0
    hand_details = []
    cnn_digit = None
    
    if results.multi_hand_landmarks and results.multi_handedness:
        mp_hands, mp_drawing, _ = initialize_mediapipe()
        
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_label = handedness.classification[0].label
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image_cv, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Count fingers
            finger_status, finger_count = count_fingers(hand_landmarks, hand_label)
            total_fingers += finger_count
            
            # Store hand details
            hand_details.append({
                'hand': hand_label,
                'count': finger_count,
                'fingers': finger_status
            })
            
            # Add text to image
            cv2.putText(image_cv, f"{hand_label} Hand: {finger_count}", 
                       (20, 50 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # CNN prediction on first hand
            if idx == 0 and model is not None:
                try:
                    # Extract hand region
                    coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                    x_min, y_min = min(x for x, y in coords), min(y for x, y in coords)
                    x_max, y_max = max(x for x, y in coords), max(y for x, y in coords)

                    pad = 20
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(w, x_max + pad)
                    y_max = min(h, y_max + pad)

                    hand_img = image_cv[y_min:y_max, x_min:x_max]
                    if hand_img.size > 0:
                        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                        hand_img = cv2.resize(hand_img, (64, 64))
                        hand_img = hand_img.reshape(1, 64, 64, 1) / 255.0
                        prediction = model.predict(hand_img, verbose=0)
                        cnn_digit = np.argmax(prediction)
                except Exception as e:
                    st.warning(f"CNN prediction error: {e}")
    
    # Add total count
    cv2.putText(image_cv, f"TOTAL: {total_fingers}", 
               (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    if cnn_digit is not None:
        cv2.putText(image_cv, f"CNN Digit: {cnn_digit}", 
                   (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Convert back to RGB
    result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    return result_image, total_fingers, hand_details, cnn_digit

# ========================
# MAIN APP
# ========================
def main():
    st.title("🤟 Hybrid Sign Language Recognition")
    st.write("✅ Detects both hands, counts fingers, and predicts CNN digit.")
    
    # Load model and mediapipe
    model = load_model()
    mp_hands, mp_drawing, hands = initialize_mediapipe()
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("Instructions")
        st.write("📸 Take a photo using the camera input")
        st.write("🖐️ Show your hand(s) with fingers clearly visible")
        st.write("🔢 Get results for finger counting and CNN digit prediction")
        st.write("")
        st.write("**Tips for best results:**")
        st.write("- Use good lighting")
        st.write("- Keep hands in frame")
        st.write("- For 5 fingers: show natural open palm")
        st.write("- For 4 fingers: tuck thumb into palm")
    
    # Camera input
    camera_input = st.camera_input("Take a picture of your hand gesture")
    
    if camera_input is not None:
        # Display original image
        image = Image.open(camera_input)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Process the image
            with st.spinner("Analyzing hand gesture..."):
                result_image, total_fingers, hand_details, cnn_digit = process_image(image, model, hands)
            
            # Display processed image
            st.image(result_image, use_column_width=True)
        
        # Results section
        st.header("📊 Detection Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Fingers", total_fingers)
        
        with col2:
            st.metric("Hands Detected", len(hand_details))
        
        with col3:
            number_words = ["Zero", "One", "Two", "Three", "Four", "Five", 
                          "Six", "Seven", "Eight", "Nine", "Ten"]
            word = number_words[total_fingers] if total_fingers < len(number_words) else str(total_fingers)
            st.metric("In Words", word)
            
        with col4:
            if cnn_digit is not None:
                st.metric("CNN Prediction", cnn_digit)
            else:
                st.metric("CNN Prediction", "N/A")
        
        # Detailed hand analysis
        if hand_details:
            st.header("🖐️ Hand Analysis")
            for i, hand in enumerate(hand_details):
                with st.expander(f"{hand['hand']} Hand - {hand['count']} fingers"):
                    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    
                    cols = st.columns(5)
                    for j, (finger, status) in enumerate(zip(finger_names, hand['fingers'])):
                        with cols[j]:
                            status_emoji = "👆" if status else "👇"
                            st.write(f"{status_emoji} {finger}")

if __name__ == "__main__":
    main()
