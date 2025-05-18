# from flask import Flask, render_template
# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

app = Flask(__name__)
model = load_model('model/sign_model.h5')
labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])
                x1, y1 = int(x_min * w), int(y_min * h)
                x2, y2 = int(x_max * w), int(y_max * h)

                roi = frame[y1:y2, x1:x2]
                if roi.size != 0:
                    roi = cv2.resize(roi, (64, 64))
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray = gray.reshape(1, 64, 64, 1) / 255.0
                    pred = model.predict(gray)
                    label = labels[np.argmax(pred)]
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

