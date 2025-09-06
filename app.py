import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.hands import HandLandmark
from PIL import Image
import tensorflow as tf
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("hand_digit_model_best.keras", compile=False)
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

@st.cache_resource
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
    return mp_hands, mp_drawing, hands

def count_fingers(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    fingers = []
    
    thumb_tip = lm[HandLandmark.THUMB_TIP]
    thumb_mcp = lm[HandLandmark.THUMB_MCP]
    
    if hand_label == "Right":
        thumb_extended = thumb_tip.x > thumb_mcp.x + 0.015
    else:
        thumb_extended = thumb_tip.x < thumb_mcp.x - 0.015
    
    fingers.append(1 if thumb_extended else 0)
    fingers.append(1 if lm[HandLandmark.INDEX_FINGER_TIP].y < lm[HandLandmark.INDEX_FINGER_PIP].y else 0)
    fingers.append(1 if lm[HandLandmark.MIDDLE_FINGER_TIP].y < lm[HandLandmark.MIDDLE_FINGER_PIP].y else 0)
    fingers.append(1 if lm[HandLandmark.RING_FINGER_TIP].y < lm[HandLandmark.RING_FINGER_PIP].y else 0)
    fingers.append(1 if lm[HandLandmark.PINKY_TIP].y < lm[HandLandmark.PINKY_PIP].y else 0)

    return fingers, sum(fingers)

def process_image(image, model, hands):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_cv = cv2.flip(image_cv, 1)
    h, w, c = image_cv.shape
    
    rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    total_fingers = 0
    hand_details = []
    cnn_digit = None
    
    if results.multi_hand_landmarks and results.multi_handedness:
        mp_hands, mp_drawing, _ = initialize_mediapipe()
        
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_label = handedness.classification[0].label
            mp_drawing.draw_landmarks(image_cv, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            finger_status, finger_count = count_fingers(hand_landmarks, hand_label)
            total_fingers += finger_count
            
            hand_details.append({'hand': hand_label, 'count': finger_count, 'fingers': finger_status})
            
            cv2.putText(image_cv, f"{hand_label} Hand: {finger_count}", 
                       (20, 50 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if idx == 0 and model is not None:
                try:
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
                except:
                    pass
    
    cv2.putText(image_cv, f"TOTAL: {total_fingers}", (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    if cnn_digit is not None:
        cv2.putText(image_cv, f"CNN: {cnn_digit}", (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return result_image, total_fingers, hand_details, cnn_digit

def main():
    st.title("🤟 Hybrid Sign Language Recognition")
    st.write("✅ Detects both hands, counts fingers, and predicts CNN digit.")
    
    model = load_model()
    mp_hands, mp_drawing, hands = initialize_mediapipe()
    
    with st.sidebar:
        st.header("Instructions")
        st.write("📸 Take a photo using camera input")
        st.write("🖐️ Show hands with fingers visible")
        st.write("🔢 Get finger count + CNN prediction")
    
    camera_input = st.camera_input("Take a picture of your hand gesture")
    
    if camera_input is not None:
        image = Image.open(camera_input)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            with st.spinner("Analyzing..."):
                result_image, total_fingers, hand_details, cnn_digit = process_image(image, model, hands)
            st.image(result_image, use_column_width=True)
        
        st.header("📊 Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Fingers", total_fingers)
        
        with col2:
            st.metric("Hands Detected", len(hand_details))
        
        with col3:
            words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
            word = words[total_fingers] if total_fingers < len(words) else str(total_fingers)
            st.metric("In Words", word)
            
        with col4:
            st.metric("CNN Prediction", cnn_digit if cnn_digit is not None else "N/A")
        
        if hand_details:
            st.header("🖐️ Hand Details")
            for hand in hand_details:
                with st.expander(f"{hand['hand']} Hand - {hand['count']} fingers"):
                    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    cols = st.columns(5)
                    for j, (finger, status) in enumerate(zip(finger_names, hand['fingers'])):
                        with cols[j]:
                            emoji = "👆" if status else "👇"
                            st.write(f"{emoji} {finger}")

if __name__ == "__main__":
    main()
