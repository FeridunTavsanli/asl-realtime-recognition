import cv2
import mediapipe as mp
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from collections import Counter

# ------------------------
# 1. MODEL
model_path = "bestModel_final_v2.keras"
model = load_model(model_path)
IMG_SIZE = 224
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["nothing", "space"]

# ------------------------
# 2. MediaPipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# ------------------------
# 2a. hand_landmarker.task kontrol ve indir
model_path_mp = "hand_landmarker.task"
if not os.path.exists(model_path_mp):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path_mp)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path_mp),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

# ------------------------
# 3. Kelime sistemi
word = ""
last_letter = ""
last_time = 0
delay = 1.0
current_letter_to_add = ""  # eklenmeyi bekleyen harf

# ------------------------
# 3a. Tahmin stabilizasyonu
pred_buffer = []
BUFFER_SIZE = 5

# ------------------------
# 4. Buton alanları
button_list = ["Add", "Delete", "Space", "Finish"]
button_coords = {}  # her butonun x1,y1,x2,y2

def draw_buttons(frame):
    h, w, _ = frame.shape
    button_w, button_h = 120, 50
    margin = 20
    for i, b in enumerate(button_list):
        x1 = margin + i*(button_w + margin)
        y1 = h - button_h - margin
        x2 = x1 + button_w
        y2 = y1 + button_h
        button_coords[b] = (x1, y1, x2, y2)
        color = (0,0,200)
        if b=="Add" and current_letter_to_add=="":
            color = (100,100,100)  # harf yoksa gri
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, -1)
        cv2.putText(frame, b, (x1+10,y1+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

def check_button_click(x, y):
    global word, current_letter_to_add
    for b, (x1,y1,x2,y2) in button_coords.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            if b=="Add" and current_letter_to_add:
                word += current_letter_to_add
                current_letter_to_add = ""
            elif b=="Delete":
                word = word[:-1]
            elif b=="Space":
                word += " "
            elif b=="Finish":
                print("Word completed:", word)
                word = ""
            break

# mouse callback
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        check_button_click(x,y)

# ------------------------
# 5. Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("ASL Realtime", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ASL Realtime", mouse_callback)
cv2.resizeWindow("ASL Realtime", 960, 720)

# CLAHE tanımla
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks]
            y_list = [lm.y for lm in hand_landmarks]
            x_center = int(np.mean(x_list) * w)
            y_center = int(np.mean(y_list) * h)
            box_size = int(max(max(x_list)-min(x_list), max(y_list)-min(y_list)) * w * 1.2)
            x_min = max(x_center - box_size // 2, 0)
            y_min = max(y_center - box_size // 2, 0)
            x_max = min(x_center + box_size // 2, w)
            y_max = min(y_center + box_size // 2, h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            crop_img = frame[y_min:y_max, x_min:x_max]
            if crop_img.size == 0:
                continue
            crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))

            # ------------------------
            # Işık düzeltme (CLAHE)
            crop_img_ycrcb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(crop_img_ycrcb)
            y = clahe.apply(y)
            crop_img_ycrcb = cv2.merge([y, cr, cb])
            crop_img = cv2.cvtColor(crop_img_ycrcb, cv2.COLOR_YCrCb2RGB)

            crop_img = crop_img / 255.0
            crop_img = np.expand_dims(crop_img, axis=0)

            # ------------------------
            # Tahmin
            pred = model.predict(crop_img, verbose=0)
            class_idx = np.argmax(pred)
            confidence = pred[0][class_idx]
            label = class_names[class_idx]

            if label=="nothing":
                continue

            if confidence>0.5:
                # buffer'a ekle
                pred_buffer.append(label)
                if len(pred_buffer) > BUFFER_SIZE:
                    pred_buffer.pop(0)
                # en sık görülen harfi al
                most_common_label = Counter(pred_buffer).most_common(1)[0][0]
                if most_common_label=="space":
                    current_letter_to_add = " "
                else:
                    current_letter_to_add = most_common_label

            cv2.putText(frame, f"{label} {confidence:.2f}", (x_min, max(30, y_min-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

    # kelime
    cv2.putText(frame, f"Word: {word}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
    # bekleyen harf
    cv2.putText(frame, f"Next: {current_letter_to_add}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 3)
    # sabit
    cv2.putText(frame, "TEST RUNNING", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    # butonları çiz
    draw_buttons(frame)

    cv2.imshow("ASL Realtime", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key==ord('r'):
        word = ""
        current_letter_to_add = ""
        pred_buffer = []

cap.release()
cv2.destroyAllWindows()     