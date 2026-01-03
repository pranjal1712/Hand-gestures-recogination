import mediapipe as mp          # pip install mediapipe  this use for hand , face and pose  detection and this google library  
import cv2                      # pip install opencv-python   this use for image processing and  OPEN CAMERA , frame showing
import csv                      # this use for save data in csv format
import os                       # this use for file path handling and folder creation



# hand landmark model basically in this mediapipe library there are many pre trained model for hand , face and pose detection ke liye
mp_hands = mp.solutions.hands                
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,     #only  one hand detection
                       # if confidence  level  kum hoga  60% se  galat detection hoga  
                       #or agar jiyada hoga toh accurate 
                       min_detection_confidence=0.6,   
                       # agar tracking confidence 60% se kam hoga toh  hand landmark tracking  band ho jayega 
                       #   otherwise tracking chalta rahega 
                       min_tracking_confidence=0.6)


# landmark and point  ko screen mai show karne ke liye 
mp_draw = mp.solutions.drawing_utils


# data set  ko " dataset" ko iss folder mai save karenge 
# or agar folder nahi hai toh create kar denge
SAVE_DIR = "dataset" 
os.makedirs(SAVE_DIR, exist_ok=True)

# only A-Z letter ke liye data collect karenge 
# #lekin agar letter in lower case mai hoga toh usko upper mai convert kar dega
# letter ko dataset wale folder mai alag alag folder mai save karenge if data sample is A so vo dataset/A folder mai save hoga
letter = input("Enter letter (A-Z): ").upper()
save_path = os.path.join(SAVE_DIR, letter)
os.makedirs(save_path, exist_ok=True)


# open webcam and start capturing frames 
# press S to save landmark data
# press Q to quit

cap = cv2.VideoCapture(0)
count = 0

print("Press S to save landmark data")
print("Press Q to quit")

# agar  press Q nahi kiya toh ye loop chalta rahega
# frame read karta rahega webcam se
# frame 

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])   # 21 points → 63 values

        if cv2.waitKey(1) & 0xFF == ord('s'):
            file = os.path.join(save_path, f"{count}.csv")
            with open(file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmarks)
            print("Saved:", file)
            count += 1

    cv2.imshow("Collect Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

