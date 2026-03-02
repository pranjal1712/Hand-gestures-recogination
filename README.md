---
title: Gesture Flow AI
emoji: 🖐️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.31.1
python_version: 3.11
app_file: app.py
pinned: false
---

# 🖐️ Gesture-Flow AI: Hand Gesture Recognition

A premium, responsive, and real-time hand gesture recognition system built with **Streamlit**, **MediaPipe**, and **TensorFlow-CPU**.

## ✨ Features
- **Real-time Detection**: Smooth hand landmark tracking using MediaPipe.
- **ASL Recognition**: Predicts hand gestures with high confidence using a custom TFLite model.
- **Sentence Builder**: Automatically constructs sentences based on stable gesture predictions.
- **Premium UI**: Sleek Dark Mode with Glassmorphism and responsive design for Mobile, Tablet, and Desktop.
- **Custom Controls**: 
  - Change Camera Index (for external or back cameras).
  - Toggle Mirror Mode.
  - Manual Edit: Space, Backspace, and Clear buttons.

---

## 🚀 How to Run (Setup Guide)

Follow these simple steps to run the project on your machine:

### 1. Prerequisites
Make sure you have **Python 3.8+** installed.

### 2. Clone the Repository
Inside your project folder, open the terminal.

### 3. Install Dependencies
Run this command to install all required libraries:
```bash
pip install -r requirements.txt
```

### 4. Launch the App
Run the following command to start the Streamlit server:
```bash
streamlit run streamlit_app.py
```

After running this, the app will automatically open in your default browser at `http://localhost:8501`.

---

## 🛠️ UI Controls
| Feature | Description |
| :--- | :--- |
| **Engage Camera** | Start or stop the live video stream. |
| **Camera Index** | Switch between multiple cameras (0 is default, 1 or 2 for external/back camera). |
| **Mirror View** | Reflect the camera feed (useful for selfie mode). |
| **Sentence Box** | Shows the text you have built using your hands! |

---

## 📝 Note for Stability
To add a letter to the sentence, hold your hand steady in that gesture for a few moments (about 5 frames). The app will automatically detect it and append it to your current word.

---

## ✋ Original Project Info
A real-time Hand Gesture Recognition System built using Python, OpenCV, MediaPipe, and Machine Learning.
The system recognizes hand gestures (A–Z / predefined gestures) from a webcam and converts them into meaningful actions or text.

### 📂 Project Structure
- `dataset/`: Collected hand landmark data
- `model/`: Trained ML model
- `analysis_outputs/`: Training graphs & analysis
- `collect_landmarks.py`: Dataset creation
- `train_classifier.py`: Model training
- `live_landmark_infer.py`: Real-time inference
- `hand_gesture_landmark_ui.py`: UI for gesture recognition (Tkinter)

---

Enjoy using **Gesture-Flow AI**! 🚀
