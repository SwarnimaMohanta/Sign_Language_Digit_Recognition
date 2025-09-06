<h1>📌 Sign Language Digit Recognition</h1>

This project is a real-time sign language digit recognition system that uses MediaPipe, OpenCV, and a deep learning model (CNN) to detect and recognize hand gestures representing digits (0–9). It provides both visual and voice feedback for accessibility and smooth interaction.<br><br>

👉 Try the live demo here:
🔗https://signlanguagedigitrecognition-livktr9ejvezjnvjm7hnnz.streamlit.app/<br><br>

<h3>🚀 Features</h3>

Real-Time Hand Detection: Uses MediaPipe Hands for accurate detection and tracking of finger positions.

Finger Counting & Gesture Detection: Counts extended fingers for quick numeric detection.

Deep Learning Model (CNN): Trained on a custom Sign Language Digits Dataset for higher recognition accuracy.

Voice Feedback: Uses pyttsx3 for instant spoken output of detected numbers.

Data Augmentation: Improves model generalization using rotation, zoom, brightness adjustment, and shifts.

Multi-Hand Support: Detects and counts fingers from both hands simultaneously.

Optimized for Smooth Performance: Non-blocking audio processing and lightweight video handling.<br><br>

<h3>🛠️ Technologies Used</h3>

Python – Core programming language

OpenCV – Real-time image capture and processing

MediaPipe – Hand landmark detection

Keras & TensorFlow – CNN model training and prediction

NumPy – Array and image data handling

scikit-learn – Dataset splitting & evaluation

pyttsx3 – Offline text-to-speech engine<br><br>

<h3>📂 Project Workflow</h3>

Model Training

Loads the Sign Language Digits Dataset.

Preprocesses and augments images.

Builds and trains a CNN with Conv2D, MaxPooling, BatchNormalization, and Dropout layers.

Saves the trained model for real-time prediction.

Real-Time Recognition

Captures webcam feed using OpenCV.

Detects hand landmarks via MediaPipe.

Counts fingers and matches gestures to digits.

Displays the recognized number on the screen.

Speaks the detected digit via text-to-speech.
