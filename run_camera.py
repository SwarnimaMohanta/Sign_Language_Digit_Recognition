import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
from mediapipe.python.solutions.hands import HandLandmark

# ========================
# VOICE ENGINE (Non-blocking)
# ========================
engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speak_async(text):
    threading.Thread(target=lambda: (engine.say(text), engine.runAndWait()), daemon=True).start()

# ========================
# MEDIAPIPE SETUP
# ========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ========================
# SIMPLE AND ACCURATE FINGER COUNTING
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
# CAMERA LOOP
# ========================
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Simple Finger Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Simple Finger Counter", 1200, 800)

# Make window fullscreen
cv2.namedWindow("Simple Finger Counter", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Simple Finger Counter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
last_total = None
frame_count = 0

print("Simple Finger Counter Started")
print("Press ESC to exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    total_fingers = 0
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            
            # Get hand label
            hand_label = handedness.classification[0].label
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Count fingers
            finger_status, finger_count = count_fingers(hand_landmarks, hand_label)
            total_fingers += finger_count
            
            # Display results for this hand
            y_pos = 50 + (idx * 200)
            
            # Hand label and count
            cv2.putText(frame, f"{hand_label} Hand: {finger_count}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Individual finger status with detailed thumb info
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            for i, (name, status) in enumerate(zip(finger_names, finger_status)):
                color = (0, 255, 0) if status else (0, 0, 255)
                status_text = "UP" if status else "DOWN"
                cv2.putText(frame, f"{name}: {status_text}", 
                           (20, y_pos + 30 + (i * 25)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show thumb position for debugging with threshold info
            thumb_tip = hand_landmarks.landmark[HandLandmark.THUMB_TIP]
            thumb_mcp = hand_landmarks.landmark[HandLandmark.THUMB_MCP]
            thumb_diff = abs(thumb_tip.x - thumb_mcp.x)
            threshold_met = thumb_diff >= 0.015
            threshold_color = (0, 255, 0) if threshold_met else (0, 0, 255)
            cv2.putText(frame, f"Thumb distance: {thumb_diff:.3f} (need: 0.015)", 
                       (20, y_pos + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, threshold_color, 1)
    
    # Display total count
    cv2.putText(frame, f"TOTAL: {total_fingers}", 
               (w//2 - 100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    
    # Instructions with updated thumb tips
    cv2.putText(frame, "For 5 fingers: Just show natural open palm", 
               (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "For 4 fingers: Tuck thumb into palm", 
               (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "Press ESC to exit", 
               (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Voice feedback (less frequent)
    if frame_count % 30 == 0:  # Every 30 frames
        if total_fingers != last_total:
            if total_fingers == 0:
                speak_async("Zero")
            elif total_fingers == 1:
                speak_async("One")
            elif total_fingers == 2:
                speak_async("Two")
            elif total_fingers == 3:
                speak_async("Three")
            elif total_fingers == 4:
                speak_async("Four")
            elif total_fingers == 5:
                speak_async("Five")
            else:
                speak_async(f"{total_fingers}")
            last_total = total_fingers
    
    cv2.imshow("Simple Finger Counter", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed successfully!")