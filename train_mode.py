import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# -------------------------
# Load dataset
# -------------------------
def load_dataset(dataset_path):
    images, labels = [], []
    for label in range(10):  # 0–9 classes
        folder = os.path.join(dataset_path, str(label))
        if not os.path.exists(folder):
            continue
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(label)
    images = np.array(images, dtype=np.float32) / 255.0
    images = images.reshape(-1, 64, 64, 1)
    labels = np.array(labels)
    return images, labels

dataset_path = r"C:\Users\SWARNIMA MOHANTA\OneDrive\Desktop\hand_digits_project\Sign-Language-Digits-Dataset-master\Dataset"
X, y = load_dataset(dataset_path)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# Data augmentation
# -------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)
datagen.fit(X_train)

# -------------------------
# Build deeper CNN
# -------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes: digits 0–9
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -------------------------
# Callbacks
# -------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("hand_digit_model_best.keras", monitor='val_accuracy', save_best_only=True)

# -------------------------
# Train model
# -------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# Save final model
model.save("hand_digit_model.keras")

print("✅ Training complete. Best model saved as 'hand_digit_model_best.keras'")
