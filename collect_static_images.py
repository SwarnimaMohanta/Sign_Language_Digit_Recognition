import cv2
import os
import time
import shutil

# --- Configuration ---
# Define the poses and the angles you want to capture for each.
POSES_TO_RECORD = {
    "thumbs_up": ["front", "side"],
    "thumbs_down": ["front", "side"],
    "fist": ["front", "top"],
    "rock_on": ["front", "back"]
}

# Define which hands to record
HANDS_TO_RECORD = ["left", "right"]

# Number of images to capture for each angle of a pose.
NUM_IMAGES_PER_ANGLE = 50

# Base path for saving the data.
BASE_DATA_PATH = "static_poses_multi_angle"

# --- Helper Function for Zoom ---
def zoom_frame(frame, zoom_factor=1.25):
    """
    Zooms into the center of a frame.
    """
    h, w, _ = frame.shape
    # Calculate the new dimensions of the crop
    crop_h, crop_w = int(h / zoom_factor), int(w / zoom_factor)
    
    # Calculate the top-left corner of the crop
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2
    
    # Perform the crop
    cropped = frame[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    # Resize the cropped frame back to the original size
    zoomed = cv2.resize(cropped, (w, h))
    
    return zoomed

# --- Main Logic ---

def collect_images():
    """
    Main function to run the static image collection process, capturing
    multiple angles for each pose, for both left and right hands.
    """
    # Clean up old data if it exists.
    if os.path.exists(BASE_DATA_PATH):
        print(f"'{BASE_DATA_PATH}' directory found. Removing old data...")
        shutil.rmtree(BASE_DATA_PATH)
    
    print(f"Creating new directory: '{BASE_DATA_PATH}'")
    os.makedirs(BASE_DATA_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    paused = False

    try:
        # Loop through each hand, then each pose, then each angle.
        for hand in HANDS_TO_RECORD:
            print("\n" + "#"*50)
            print(f"### GET READY TO RECORD FOR YOUR {hand.upper()} HAND ###")
            print("#"*50)
            time.sleep(5)

            for pose, angles in POSES_TO_RECORD.items():
                for angle in angles:
                    # Create the final directory path, e.g., static_poses_multi_angle/left/thumbs_up/front
                    final_path = os.path.join(BASE_DATA_PATH, hand, pose, angle)
                    os.makedirs(final_path, exist_ok=True)
                    
                    print("\n" + "="*50)
                    print(f"GET READY: '{pose.upper()}' ({angle.upper()}) - {hand.upper()} HAND")
                    print("="*50)

                    # Give the user a countdown to get ready.
                    for t in range(10, 0, -1):
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        display_frame = zoom_frame(frame)
                        display_frame = cv2.flip(display_frame, 1)
                        
                        prompt_text = f"POSE: {pose.upper()} ({angle.upper()}) - {hand.upper()} HAND"
                        countdown_text = f"GET READY: {t}"
                        cv2.putText(display_frame, prompt_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(display_frame, countdown_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow('Data Collection', display_frame)
                        
                        key = cv2.waitKey(1000)
                        if key == 32: # Spacebar to pause/play
                            paused = not paused
                        
                        while paused:
                            pause_frame = display_frame.copy()
                            cv2.putText(pause_frame, "PAUSED", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                            cv2.imshow('Data Collection', pause_frame)
                            key = cv2.waitKey(100)
                            if key == 32:
                                paused = not paused


                    print(f"\n--- Capturing {NUM_IMAGES_PER_ANGLE} images ---")
                    
                    # Capture the images.
                    for i in range(NUM_IMAGES_PER_ANGLE):
                        ret, frame = cap.read()
                        if not ret:
                            continue

                        display_frame = zoom_frame(frame)
                        display_frame = cv2.flip(display_frame, 1)
                        cv2.putText(display_frame, f"CAPTURING... ({i+1}/{NUM_IMAGES_PER_ANGLE})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('Data Collection', display_frame)
                        
                        image_path = os.path.join(final_path, f"{i}.jpg")
                        cv2.imwrite(image_path, frame)
                        
                        key = cv2.waitKey(100)
                        if key == 32:
                            paused = not paused
                        
                        while paused:
                            pause_frame = display_frame.copy()
                            cv2.putText(pause_frame, "PAUSED", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                            cv2.imshow('Data Collection', pause_frame)
                            key = cv2.waitKey(100)
                            if key == 32:
                                paused = not paused

                    print(f"Finished capturing for '{pose}' ({angle}) - {hand} hand.")
                    time.sleep(3)

    finally:
        # Release the webcam and close all OpenCV windows.
        print("\nData collection complete!")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_images()
