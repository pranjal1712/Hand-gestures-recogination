from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# Load model
interpreter = Interpreter("model/asl_landmark_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = np.load("model/label_map.npy", allow_pickle=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return jsonify({"label": "No hand"})

    landmarks = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    vec = np.array(landmarks, dtype=np.float32).reshape(1, 63)

    interpreter.set_tensor(input_details[0]['index'], vec)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    pred = labels[np.argmax(output)]
    return jsonify({"label": str(pred)})

if __name__ == "__main__":
    app.run(debug=True)
