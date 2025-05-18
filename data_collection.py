import cv2
import os
import time
import numpy as np
import mediapipe as mp

# Label input
label = input("Enter label (e.g. A, B, 0, 1): ").upper()
save_path = f"../dataset/{label}"
os.makedirs(save_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

print("Starting collection...")

count = 0
TARGET_SIZE = 300  # Size of the square image

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get bounding box of the hand
            x_list = [int(lm.x * w) for lm in handLms.landmark]
            y_list = [int(lm.y * h) for lm in handLms.landmark]
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)

            # Add some padding
            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            # Crop and resize
            cropped = frame[y_min:y_max, x_min:x_max]
            try:
                white_bg = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
                hand_img = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))
                white_bg[:hand_img.shape[0], :hand_img.shape[1]] = hand_img

                gray = cv2.cvtColor(white_bg, cv2.COLOR_BGR2GRAY)
                final = cv2.resize(gray, (64, 64))
                cv2.imwrite(f"{save_path}/{count}.jpg", final)
                count += 1
            except:
                pass

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.putText(frame, f"Images Collected: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Data Collection - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 1200:
        break

cap.release()
cv2.destroyAllWindows()


