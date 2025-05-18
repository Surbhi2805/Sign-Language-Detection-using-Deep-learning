
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import mediapipe as mp
import pyttsx3
import enchant

# Load model and labels
model = load_model("../model/sign_model.h5")
labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Text-to-speech
engine = pyttsx3.init()
dictionary = enchant.Dict("en_US")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# GUI setup
root = tk.Tk()
root.title("Sign Language to Text & Speech")
root.geometry("900x650")
root.configure(bg="white")

sentence = ""
current_word = ""
last_prediction = ""
stable_count = 0
STABLE_THRESHOLD = 15
confidence_threshold = 0.85 

def speak_text():
    engine.say(sentence)
    engine.runAndWait()

def clear_text():
    global sentence, current_word
    sentence = ""
    current_word = ""
    text_display.config(text="")

def backspace():
    global sentence, current_word
    sentence = sentence.strip()
    if sentence:
        sentence = sentence[:-1]
    current_word = sentence.split(" ")[-1] if " " in sentence else sentence
    text_display.config(text=sentence)
    update_suggestions()

def select_word(word):
    global sentence, current_word
    if sentence.endswith(current_word):
        sentence = sentence[:-len(current_word)]
    sentence += word + " "
    current_word = ""
    text_display.config(text=sentence)
    update_suggestions()

def add_space():
    global sentence, current_word
    sentence += " "
    current_word = ""
    text_display.config(text=sentence)
    update_suggestions()

def confirm_character():
    global sentence, current_word, last_prediction
    if last_prediction:
        sentence += last_prediction
        current_word += last_prediction
        text_display.config(text=sentence)
        update_suggestions()

def update_suggestions():
    suggestions = dictionary.suggest(current_word)
    for i in range(3):
        if i < len(suggestions):
            suggestion_btns[i].config(text=suggestions[i], state=tk.NORMAL, command=lambda w=suggestions[i]: select_word(w))
        else:
            suggestion_btns[i].config(text="...", state=tk.DISABLED)

# Video Frame
video_label = tk.Label(root)
video_label.pack()

# Sentence Display
text_display = tk.Label(root, text="", font=("Helvetica", 24), bg="white")
text_display.pack(pady=10)

# Control Buttons
btn_frame = tk.Frame(root, bg="white")
btn_frame.pack()

tk.Button(btn_frame, text="Speak", command=speak_text, font=("Helvetica", 14), width=10).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Backspace", command=backspace, font=("Helvetica", 14), width=10).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Clear", command=clear_text, font=("Helvetica", 14), width=10).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Space", command=add_space, font=("Helvetica", 14), width=10).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Add Letter", command=confirm_character, font=("Helvetica", 14), width=12).pack(side=tk.LEFT, padx=5)

# Word Suggestions
suggestion_btns = []
suggestion_frame = tk.Frame(root, bg="white")
suggestion_frame.pack(pady=10)
for _ in range(3):
    btn = tk.Button(suggestion_frame, text="...", font=("Helvetica", 14), state=tk.DISABLED, width=15)
    btn.pack(side=tk.LEFT, padx=5)
    suggestion_btns.append(btn)

# Video Capture
cap = cv2.VideoCapture(0)

def detect():
    global last_prediction, stable_count

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)  # Mirror view
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    display_label = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            x, y = max(0, int(x_min * w)), max(0, int(y_min * h))
            x2, y2 = min(w, int(x_max * w)), min(h, int(y_max * h))

            roi = frame[y:y2, x:x2]
            if roi.size != 0:
                roi_gray = cv2.cvtColor(cv2.resize(roi, (64, 64)), cv2.COLOR_BGR2GRAY)
                roi_gray = roi_gray.reshape(1, 64, 64, 1) / 255.0
                pred = model.predict(roi_gray)
                pred_confidence = np.max(pred)  # Confidence score
                label = labels[np.argmax(pred)]
                # display_label = label

            if pred_confidence > confidence_threshold:
                if label == last_prediction:
                    stable_count += 1
                else:
                    last_prediction = label
                    stable_count = 1

                # Show result when stable
                if stable_count >= STABLE_THRESHOLD:
                    cv2.putText(frame, f"Stable: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Detecting: {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    root.after(10, detect)

# Start detection loop
root.after(10, detect)
# detect()
root.mainloop()

