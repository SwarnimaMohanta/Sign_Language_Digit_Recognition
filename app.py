import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.hands import HandLandmark

# ========================
# Load CNN model
# ========================
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("hand_digit_model_best.keras")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Try loading without compiling
        try:
            return tf.keras.models.load_model("hand_digit_model_best.keras", compile=False)
        except Exception as e2:
            st.error(f"Failed to load model even without compiling: {str(e2)}")
            st.stop()

# ========================
# Mediapipe setup
# ========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ========================
# Finger counting
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

    return sum(fingers)

# ========================
# Streamlit UI
# ========================
st.title("🤟 Hybrid Sign Language Recognition")
st.markdown("✅ Detects both hands, counts fingers, and predicts CNN digit.")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        total_fingers = 0
        counts = {"Left": 0, "Right": 0}
        cnn_digit = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                hand_label = handedness.classification[0].label
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Finger counting
                finger_count = count_fingers(hand_landmarks, hand_label)
                counts[hand_label] = finger_count
                total_fingers += finger_count

                cv2.putText(frame, f"{hand_label} Hand: {finger_count}", (10, 40 + idx*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # CNN prediction (on first hand only)
                if idx == 0:
                    coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                    x_min, y_min = min(x for x, y in coords), min(y for x, y in coords)
                    x_max, y_max = max(x for x, y in coords), max(y for x, y in coords)

                    pad = 20
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(w, x_max + pad)
                    y_max = min(h, y_max + pad)

                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size > 0:
                        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                        hand_img = cv2.resize(hand_img, (64, 64))
                        hand_img = hand_img.reshape(1, 64, 64, 1) / 255.0
                        prediction = model.predict(hand_img)
                        cnn_digit = np.argmax(prediction)

        # Show totals
        cv2.putText(frame, f"TOTAL Fingers: {total_fingers}", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        if cnn_digit is not None:
            cv2.putText(frame, f"CNN Digit: {cnn_digit}", (10, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

