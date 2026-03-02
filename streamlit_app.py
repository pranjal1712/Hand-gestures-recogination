import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# --- Configuration ---
MODEL_PATH = "model/asl_landmark_model.tflite"
LABEL_PATH = "model/label_map.npy"
STABILITY_THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.8
COOLDOWN_TIME = 1.0  # seconds

# --- Session State Initialization ---
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'last_gesture' not in st.session_state:
    st.session_state.last_gesture = None
if 'stable_count' not in st.session_state:
    st.session_state.stable_count = 0
if 'last_append_time' not in st.session_state:
    st.session_state.last_append_time = 0

# --- Page Config ---
st.set_page_config(
    page_title="ASL Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide"
)

# --- Premium Dark CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(2rem, 5vw, 3.5rem);
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    
    .sentence-box {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid #4CAF50;
        padding: 20px;
        border-radius: 15px;
        min-height: 100px;
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        color: #92FE9D;
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 0 20px rgba(76, 175, 80, 0.2);
        word-wrap: break-word;
    }
    
    .prediction-card {
        text-align: center;
        padding: clamp(10px, 2vw, 15px);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.03);
        border-left: 5px solid #00C9FF;
    }
    
    .gesture-label {
        font-size: clamp(2rem, 5vw, 3rem);
        color: #00C9FF;
        font-weight: bold;
    }
    
    .confidence-value {
        font-size: clamp(1rem, 3vw, 1.5rem);
        color: #888;
    }
    
    /* Responsive Adjustments */
    @media (max-width: 768px) {
        .glass-container {
            padding: 15px;
        }
        .main-title {
            margin-bottom: 1rem;
        }
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 10px;
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Resources ---
@st.cache_resource
def load_resources():
    try:
        interpreter = Interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        labels = np.load(LABEL_PATH, allow_pickle=True)
        
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        
        return interpreter, input_details, output_details, labels, hands, mp_drawing, mp_hands
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return None

resources = load_resources()
if resources:
    interpreter, input_details, output_details, labels, hands, mp_drawing, mp_hands = resources

# --- Prediction Logic ---
def get_prediction(landmarks):
    vec = []
    for lm in landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z])
    inp = np.array(vec, dtype=np.float32).reshape(1, -1)
    
    interpreter.set_tensor(input_details["index"], inp)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details["index"])[0]
    
    idx = np.argmax(probs)
    return labels[idx], float(probs[idx])

# --- Main UI ---
st.markdown('<div class="main-title">🖐️ GESTURE-FLOW AI</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1])

with col1:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📽️ NEURAL VISION")
    
    # Sidebar or Sidebar-like options for Camera
    cam_col1, cam_col2, cam_col3 = st.columns([1, 1, 1])
    with cam_col1:
        run_camera = st.checkbox("Engage Camera", value=True)
    with cam_col2:
        camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
    with cam_col3:
        mirror_video = st.toggle("Mirror View", value=True)
        
    frame_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📊 REAL-TIME INFERENCE")
    
    pred_col, conf_col = st.columns(2)
    with pred_col:
        st.markdown('<div class="prediction-card">ID</div>', unsafe_allow_html=True)
        gesture_ui = st.empty()
    with conf_col:
        st.markdown('<div class="prediction-card">CONF</div>', unsafe_allow_html=True)
        confidence_ui = st.empty()
    
    st.markdown("---")
    st.markdown("### ✏️ SENTENCE BUILDER")
    
    c1, c2, c3 = st.columns(3)
    if c1.button("Space ␣"):
        st.session_state.sentence += " "
    if c2.button("Backspace"):
        st.session_state.sentence = st.session_state.sentence[:-1]
    if c3.button("Clear 🗑️", type="primary"):
        st.session_state.sentence = ""
        
    sentence_ui = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# --- Processing Loop ---
if run_camera and resources:
    cap = cv2.VideoCapture(int(camera_index))
    
    while run_camera:
        ret, frame = cap.read()
        if not ret: 
            st.error(f"Failed to access Camera Index {camera_index}. Try another index.")
            break
        
        if mirror_video:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        current_gesture = "NONE"
        current_conf = 0.0
        
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                current_gesture, current_conf = get_prediction(hand_lms)
                
                # Stability and Sentence Building Logic
                if current_conf > CONFIDENCE_THRESHOLD:
                    if current_gesture == st.session_state.last_gesture:
                        st.session_state.stable_count += 1
                    else:
                        st.session_state.last_gesture = current_gesture
                        st.session_state.stable_count = 1
                    
                    # Logic to append
                    if st.session_state.stable_count >= STABILITY_THRESHOLD:
                        now = time.time()
                        if now - st.session_state.last_append_time > COOLDOWN_TIME:
                            char_to_add = current_gesture.upper()
                            if char_to_add == "SPACE": char_to_add = " "
                            
                            st.session_state.sentence += char_to_add
                            st.session_state.last_append_time = now
                            st.session_state.stable_count = 0 # Reset stability after add
                else:
                    st.session_state.stable_count = 0

        # Update Video
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # Update UI Elements
        gesture_ui.markdown(f'<div class="gesture-label">{current_gesture}</div>', unsafe_allow_html=True)
        confidence_ui.markdown(f'<div class="confidence-value">{current_conf:.1%}</div>', unsafe_allow_html=True)
        sentence_ui.markdown(f'<div class="sentence-box">{st.session_state.sentence}</div>', unsafe_allow_html=True)
        
        time.sleep(0.01)
        
    cap.release()
else:
    st.info("System Standby. Active Camera to proceed.")
