import cv2
import mediapipe as mp
import pandas as pd
import pickle
import time
import pywhatkit as pwk
import threading

# ---------------- CONFIG ----------------
MODEL_FILE = "gesture_model.pkl"  # Your trained gesture model
NUM_FEATURES = 63
LABEL = "SOS_Gesture"  # Label in your model for SOS
ALERT_COOLDOWN = 10  # Seconds between alerts
USER_NAME = "Sandhiya"
WHATSAPP_NUMBER = "+919842835780"  # Recipient WhatsApp number
ADDRESS = "123, Your Street, Your City, Your Country"  # Fixed address to send
# ---------------------------------------

# ---------------- WHATSAPP ALERT ----------------
alert_lock = threading.Lock()

def send_whatsapp_alert():
    message = f"""
🚨 SOS ALERT 🚨
User: {USER_NAME}
Status: Emergency Detected!
Address: {ADDRESS}
"""
    try:
        # Send instantly using pywhatkit
        pwk.sendwhatmsg_instantly(
            WHATSAPP_NUMBER, message, wait_time=10, tab_close=True, close_time=5
        )
        print("[INFO] WhatsApp SOS Alert Sent!")
    except Exception as e:
        print(f"[ERROR] WhatsApp alert failed: {e}")

# ---------------- LOAD GESTURE MODEL ----------------
with open(MODEL_FILE, 'rb') as f:
    clf = pickle.load(f)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

last_alert_time = 0

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == NUM_FEATURES:
            X_pred = pd.DataFrame([landmarks], columns=[f'f{i}' for i in range(NUM_FEATURES)])
            pred = clf.predict(X_pred)[0]

            cv2.putText(frame, f"Gesture: {pred}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            current_time = time.time()
            if pred == LABEL and (current_time - last_alert_time > ALERT_COOLDOWN):
                with alert_lock:
                    print("[ALERT] SOS Gesture Detected!")
                    threading.Thread(target=send_whatsapp_alert).start()
                    last_alert_time = current_time

    cv2.imshow("SOS Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
