
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

DATA_DIR = "../dataset/"
labels = sorted(os.listdir(DATA_DIR))
images, y = [], []

for idx, label in enumerate(labels):
    for img_name in os.listdir(os.path.join(DATA_DIR, label)):
        img_path = os.path.join(DATA_DIR, label, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        images.append(img)
        y.append(idx)

X = np.array(images).reshape(-1, 64, 64, 1) / 255.0
y = to_categorical(np.array(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model with Dropout and Increased Depth
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# Save training history
with open('../model/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
model.save("../model/sign_model.h5")

