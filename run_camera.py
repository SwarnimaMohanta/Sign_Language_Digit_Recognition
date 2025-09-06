import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.hands import HandLandmark
from PIL import Image
import tensorflow as tf

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
        static_image_mode=True,  # Changed to True for single image processing
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    return mp_hands, mp_drawing, hands

mp_hands, mp_drawing, hands = initialize_mediapipe()

# ========================
# FINGER COUNTING FUNCTION
# ========================
def count_fingers(hand_landmarks, hand_label):
    """
    Fixed finger counting with more sensitive thumb detection
    """
    lm = hand_landmarks.landmark
    
    fingers = []
    
    # THUMB detection (more sensitive for natural 5-finger pose)
    thumb_tip = lm[HandLandmark.THUMB_TIP]
    thumb_mcp = lm[HandLandmark.THUMB_MCP]
    
    # More sensitive threshold for natural hand positions
    if hand_label == "Right":
        # Right hand: thumb extended if tip is right of MCP (reduced threshold)
        thumb_extended = thumb_tip.x > thumb_mcp.x + 0.015  # Reduced from 0.04 to 0.015
    else:
        # Left hand: thumb extended if tip is left of MCP  
        thumb_extended = thumb_tip.x < thumb_mcp.x - 0.015  # Reduced from 0.04 to 0.015
    
    fingers.append(1 if thumb_extended else 0)
    
    # INDEX finger
    if lm[HandLandmark.INDEX_FINGER_TIP].y < lm[HandLandmark.INDEX_FINGER_PIP].y:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # MIDDLE finger
    if lm[HandLandmark.MIDDLE_FINGER_TIP].y < lm[HandLandmark.MIDDLE_FINGER_PIP].y:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # RING finger
    if lm[HandLandmark.RING_FINGER_TIP].y < lm[HandLandmark.RING_FINGER_PIP].y:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # PINKY finger
    if lm[HandLandmark.PINKY_TIP].y < lm[HandLandmark.PINKY_PIP].y:
        fingers.append(1)
    else:
        fingers.append(0)
    
    return fingers, sum(fingers)

# ========================
# IMAGE PROCESSING FUNCTION
# ========================
def process_image(image):
    """Process uploaded image for sign language recognition"""
    # Convert PIL to OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Flip image for mirror effect
    image_cv = cv2.flip(image_cv, 1)
    h, w, c = image_cv.shape
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    total_fingers = 0
    hand_details = []
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            
            # Get hand label
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
            y_pos = 50 + (idx * 200)
            cv2.putText(image_cv, f"{hand_label} Hand: {finger_count}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Individual finger status
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            for i, (name, status) in enumerate(zip(finger_names, finger_status)):
                color = (0, 255, 0) if status else (0, 0, 255)
                status_text = "UP" if status else "DOWN"
                cv2.putText(image_cv, f"{name}: {status_text}", 
                           (20, y_pos + 30 + (i * 25)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add total count
    cv2.putText(image_cv, f"TOTAL: {total_fingers}", 
               (w//2 - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    
    # Convert back to RGB for Streamlit
    result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    return result_image, total_fingers, hand_details

# ========================
# MAIN APP
# ========================
def main():
    st.title("🤟 Hybrid Sign Language Recognition")
    st.write("✅ Detects both hands, counts fingers, and predicts CNN digit.")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Model failed to load. Please check the model file.")
        return
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("Instructions")
        st.write("📸 **Take a photo** using the camera input")
        st.write("🖐️ **Show your hand(s)** with fingers clearly visible")
        st.write("🔢 **Get results** for finger counting and digit prediction")
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
                result_image, total_fingers, hand_details = process_image(image)
            
            # Display processed image
            st.image(result_image, use_column_width=True)
        
        # Results section
        st.header("📊 Detection Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Fingers", total_fingers)
        
        with col2:
            st.metric("Hands Detected", len(hand_details))
        
        with col3:
            # Convert number to word
            number_words = ["Zero", "One", "Two", "Three", "Four", "Five", 
                          "Six", "Seven", "Eight", "Nine", "Ten"]
            word = number_words[total_fingers] if total_fingers < len(number_words) else str(total_fingers)
            st.metric("In Words", word)
        
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
        
        # CNN Prediction (if you want to add this)
        if st.button("🤖 Get CNN Prediction"):
            with st.spinner("Running CNN prediction..."):
                try:
                    # Preprocess image for CNN
                    img_array = np.array(image.resize((64, 64)))  # Adjust size as needed
                    img_array = img_array.astype('float32') / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Make prediction
                    prediction = model.predict(img_array)
                    predicted_digit = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("CNN Prediction", predicted_digit)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                except Exception as e:
                    st.error(f"CNN prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
