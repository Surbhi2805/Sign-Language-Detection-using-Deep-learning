import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

model = load_model("../model/sign_model.h5")
labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            h, w, _ = frame.shape
            x, y = int(x_min * w), int(y_min * h)
            x2, y2 = int(x_max * w), int(y_max * h)

            roi = frame[y:y2, x:x2]
            if roi.size != 0:
                roi = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))
                roi = roi.reshape(1, 64, 64, 1) / 255.0
                pred = model.predict(roi)
                label = labels[np.argmax(pred)]
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.rectangle(frame, (x, y), (x2, y2), (255,0,0), 2)

    cv2.imshow("Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
