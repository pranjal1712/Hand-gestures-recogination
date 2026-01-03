# hand_gesture_landmark_ui.py

import threading
import time
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2

# Try to import tflite interpreter
try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite_runtime.Interpreter")
except Exception:
    from tensorflow.lite.python.interpreter import Interpreter
    print("Using tensorflow.lite.Interpreter")

# MediaPipe
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- CONFIG ----------
MODEL_PATH = "model/asl_landmark_model.tflite"
LABEL_MAP_PATH = "model/label_map.npy"

FRAME_FLIP = True           # mirror preview
CAMERA_INDEX = 0

append_confidence_threshold = 0.80
stable_frames = 6
append_cooldown = 0.8       # seconds
MP_MIN_DET = 0.6
MP_MIN_TRACK = 0.6
MP_MAX_NUM_HANDS = 1
# ----------------------------


class LandmarkModel:
    def __init__(self, model_path, label_path):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not Path(label_path).exists():
            raise FileNotFoundError(f"Label map not found: {label_path}")

        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()

        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]

        self.input_index = inp["index"]
        self.output_index = out["index"]
        self.input_shape = inp["shape"]     # [1,63]
        self.input_dtype = inp["dtype"]

        self.output_scale, self.output_zero_point = out.get("quantization", (0.0, 0))

        # load labels
        self.labels = np.load(label_path, allow_pickle=True)
        print("Loaded labels:", self.labels)

        print("Model loaded. Input shape:", self.input_shape, "dtype:", self.input_dtype)

    def predict(self, vec63: np.ndarray):
        """
        vec63: 1D vector length 63 (x,y,z * 21)
        returns: (label_str, probs)
        """
        if vec63.shape != (63,):
            raise ValueError(f"Expected shape (63,), got {vec63.shape}")

        # to [1,63]
        inp = vec63.astype(np.float32).reshape(1, -1)

        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)

        probs = out[0]

        # dequantize if needed
        if self.output_scale and self.output_scale != 0:
            probs = self.output_scale * (probs.astype(np.float32) - self.output_zero_point)

        # softmax safety
        if probs.max() > 1.0 or probs.min() < 0.0:
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)

        best = int(np.argmax(probs))
        label = str(self.labels[best])
        return label, probs


class App:
    def __init__(self, root):
        self.root = root
        root.title("ASL Hand Gesture — Landmark UI")

        # Video display
        self.video_label = tk.Label(root)
        self.video_label.pack()

        # Control frame
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.X, pady=6)

        tk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=4)
        self.model_path_var = tk.StringVar(value=str(MODEL_PATH))
        tk.Entry(control_frame, textvariable=self.model_path_var, width=40)\
            .pack(side=tk.LEFT, padx=4)

        self.start_btn = tk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=4)

        self.stop_btn = tk.Button(control_frame, text="Stop Camera",
                                  command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)

        self.detect_btn = tk.Button(control_frame, text="Start Detection",
                                    command=self.toggle_detection, state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=4)

        # Word controls
        word_frame = tk.Frame(root)
        word_frame.pack(fill=tk.X, pady=4)
        tk.Button(word_frame, text="Space", command=self.add_space).pack(side=tk.LEFT, padx=4)
        tk.Button(word_frame, text="Backspace", command=self.backspace).pack(side=tk.LEFT, padx=4)
        tk.Button(word_frame, text="Clear Word", command=self.clear_word).pack(side=tk.LEFT, padx=4)

        # Prediction & word labels
        self.pred_label = tk.Label(root, text="Prediction: —", font=("Helvetica", 20))
        self.pred_label.pack(pady=6)

        self.word_label = tk.Label(root, text="Word: ", font=("Helvetica", 26, "bold"))
        self.word_label.pack(pady=6)

        self.status_label = tk.Label(root, text="Status: Idle")
        self.status_label.pack()

        # State
        self.cap = None
        self.running = False
        self.detecting = False
        self.thread = None
        self._imgtk = None

        self.model = None
        self.hands = None

        # word-building
        self.current_word = ""
        self._last_stable = None
        self._stable_count = 0
        self._last_append_time = 0.0

        self.append_confidence_threshold = append_confidence_threshold
        self.stable_frames = stable_frames
        self.append_cooldown = append_cooldown

        # auto-load model if path exists
        if Path(MODEL_PATH).exists() and Path(LABEL_MAP_PATH).exists():
            try:
                self._load_model()
            except Exception as e:
                print("Auto model load failed:", e)

    def _load_model(self):
        path = self.model_path_var.get()
        label_path = LABEL_MAP_PATH
        self.model = LandmarkModel(path, label_path)
        self.detect_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Model loaded")
        messagebox.showinfo("Model",
                            f"Loaded model: {os.path.basename(path)}\n"
                            f"Input shape: {self.model.input_shape}")

    def start_camera(self):
        if self.running:
            return

        # if model is not loaded, try load
        if self.model is None:
            try:
                self._load_model()
            except Exception as e:
                messagebox.showerror("Model error", str(e))
                return

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open webcam (index {CAMERA_INDEX}).")
            return

        # Mediapipe hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_NUM_HANDS,
            min_detection_confidence=MP_MIN_DET,
            min_tracking_confidence=MP_MIN_TRACK
        )

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.detect_btn.config(state=tk.NORMAL if self.model else tk.DISABLED)

        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        self.status_label.config(text="Status: Camera started")

    def stop_camera(self):
        if not self.running:
            return
        self.running = False
        time.sleep(0.2)

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.hands:
            try:
                self.hands.close()
            except Exception:
                pass
            self.hands = None

        self.video_label.config(image="")
        self._imgtk = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Camera stopped")

    def toggle_detection(self):
        if not self.model:
            messagebox.showwarning("Model", "Load a model first.")
            return
        self.detecting = not self.detecting
        self.detect_btn.config(text="Stop Detection" if self.detecting else "Start Detection")
        self.status_label.config(text="Status: Detecting" if self.detecting else "Status: Model loaded")

        self._last_stable = None
        self._stable_count = 0

    # --- word editing buttons ---
    def add_space(self):
        self.current_word += " "
        self._update_word_label()

    def backspace(self):
        self.current_word = self.current_word[:-1]
        self._update_word_label()

    def clear_word(self):
        self.current_word = ""
        self._update_word_label()

    def _update_word_label(self):
        self.word_label.config(text=f"Word: {self.current_word}")

    # --- prediction processing (AUTO word build) ---
    def _process_prediction(self, label: str, probs: np.ndarray):
        """
        label: e.g. 'A', 'B', ..., 'SPACE'
        probs: class probabilities
        """
        conf = float(np.max(probs))
        self.pred_label.config(text=f"Prediction: {label}  ({conf:.2f})")

        # confidence low -> ignore
        if conf < self.append_confidence_threshold:
            self._last_stable = None
            self._stable_count = 0
            return

        now = time.time()

        # handle SPACE class
        if label.upper() == "SPACE":
            # treat as word space
            if now - self._last_append_time > self.append_cooldown:
                self.current_word += " "
                self._update_word_label()
                self._last_append_time = now
            # space pe stability track nahi karna
            self._last_stable = None
            self._stable_count = 0
            return

        # only single alphabetic letters for word
        if not (len(label) == 1 and label.isalpha()):
            return

        # stable detection
        if self._last_stable == label:
            self._stable_count += 1
        else:
            self._last_stable = label
            self._stable_count = 1

        if self._stable_count >= self.stable_frames:
            if now - self._last_append_time > self.append_cooldown:
                self.current_word += label.upper()
                self._update_word_label()
                self._last_append_time = now
            self._stable_count = 0

    # --- camera loop ---
    def _camera_loop(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            if FRAME_FLIP:
                frame_disp = cv2.flip(frame, 1)
            else:
                frame_disp = frame.copy()

            # mediapipe expects RGB
            img_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            hand_vec = None

            if self.hands:
                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]

                    # draw landmarks
                    mp_drawing.draw_landmarks(
                        frame_disp, hand_lms, mp_hands.HAND_CONNECTIONS
                    )

                    # 63-dim vector (x,y,z) * 21
                    vals = []
                    for lm in hand_lms.landmark:
                        vals.extend([lm.x, lm.y, lm.z])
                    hand_vec = np.array(vals, dtype=np.float32)

            # show frame
            img_rgb_disp = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb_disp)
            img_resized = img_pil.resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img_resized)
            self._imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.video_label.imgtk = imgtk

            # prediction if detecting & we have landmarks
            if self.detecting and self.model and hand_vec is not None:
                try:
                    label, probs = self.model.predict(hand_vec)
                    self._process_prediction(label, probs)
                except Exception as e:
                    print("Inference error:", e)
                    self.pred_label.config(text="Prediction: Error")
            else:
                if self.detecting:
                    self.pred_label.config(text="Prediction: (no hand)")

            time.sleep(0.02)

        self.running = False


def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
