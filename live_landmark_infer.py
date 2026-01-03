import cv2
import numpy as np
import mediapipe as mp

# TFLite interpreter import
try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime.Interpreter")
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter
    print("Using tensorflow.lite.Interpreter")

MODEL_PATH = "model/asl_landmark_model.tflite"
LABEL_PATH = "model/label_map.npy"

# Load labels
LABELS = np.load(LABEL_PATH, allow_pickle=True)

print("Labels:", LABELS)

# Load TFLite model
interpreter = Interpreter(MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
output_index = output_details["index"]

print("Input shape:", input_details["shape"])
print("Output shape:", output_details["shape"])

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Mediapipe wants RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction_text = "No hand"

    if result.multi_hand_landmarks:
        hand_lms = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # 21 * (x,y,z) = 63 features
        vec = []
        for lm in hand_lms.landmark:
            vec.extend([lm.x, lm.y, lm.z])

        # Convert to numpy
        inp = np.array(vec, dtype=np.float32).reshape(1, -1)

        # Run TFLite
        interpreter.set_tensor(input_index, inp)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_index)[0]

        best_idx = int(np.argmax(probs))
        best_label = LABELS[best_idx]
        best_conf = float(probs[best_idx])

        # Top-3 debug (optional)
        top3_idx = np.argsort(probs)[-3:][::-1]
        debug_str = ", ".join(
            [f"{LABELS[i]}:{probs[i]:.2f}" for i in top3_idx]
        )
        # print(debug_str)

        prediction_text = f"{best_label} ({best_conf:.2f})"

        # Draw box + text
        cv2.rectangle(frame, (10, 10), (350, 60), (0, 0, 0), -1)
        cv2.putText(frame, "Pred: " + prediction_text,
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    else:
        cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, "No hand",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)

    cv2.imshow("ASL Landmark Live", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
