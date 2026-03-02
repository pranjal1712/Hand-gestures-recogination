import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import threading

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

# --- Page Config ---
st.set_page_config(
    page_title="ASL Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide"
)

# --- Session State ---
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

# --- CSS Styling ---
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

    .prediction-panel {
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border-left: 5px solid #00C9FF;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model_and_labels():
    interpreter = Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    labels = np.load(LABEL_PATH, allow_pickle=True)
    return interpreter, input_details, output_details, labels

model_resources = load_model_and_labels()
interpreter, input_details, output_details, labels = model_resources

# --- Mediapipe Setup ---
# Using explicit submodule imports to avoid 'solutions' attribute error
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# --- Shared State for WebRTC ---
class SharedState:
    def __init__(self):
        self.gesture = "NONE"
        self.confidence = 0.0
        self.last_appended_gesture = None
        self.stable_count = 0
        self.last_append_time = 0
        self.lock = threading.Lock()

shared_state = SharedState()

# --- WebRTC Processor ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process Frame
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        local_gesture = "NONE"
        local_conf = 0.0
        
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # Predict
                vec = []
                for lm in hand_lms.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
                inp = np.array(vec, dtype=np.float32).reshape(1, -1)
                
                interpreter.set_tensor(input_details["index"], inp)
                interpreter.invoke()
                probs = interpreter.get_tensor(output_details["index"])[0]
                idx = np.argmax(probs)
                local_gesture = labels[idx]
                local_conf = float(probs[idx])

        # Update Shared State
        with shared_state.lock:
            shared_state.gesture = local_gesture
            shared_state.confidence = local_conf
            
            # Sentence Building Logic inside Processor (Thread Safe)
            if local_conf > CONFIDENCE_THRESHOLD:
                if local_gesture == shared_state.last_appended_gesture:
                    shared_state.stable_count += 1
                else:
                    shared_state.last_appended_gesture = local_gesture
                    shared_state.stable_count = 1
                
                if shared_state.stable_count >= STABILITY_THRESHOLD:
                    now = time.time()
                    if now - shared_state.last_append_time > COOLDOWN_TIME:
                        char = local_gesture.upper()
                        if char == "SPACE": char = " "
                        st.session_state.sentence += char
                        shared_state.last_append_time = now
                        shared_state.stable_count = 0
            else:
                shared_state.stable_count = 0

        return frame.from_ndarray(img, format="bgr24")

# --- Main UI ---
st.markdown('<div class="main-title">🖐️ GESTURE-FLOW AI</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 1])

with col1:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📽️ NEURAL VISION (Cloud Camera)")
    
    webrtc_ctx = webrtc_streamer(
        key="gesture-recognition",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False}
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📊 STATUS")
    
    gesture_placeholder = st.empty()
    confidence_placeholder = st.empty()
    
    st.markdown("---")
    st.markdown("### ✏️ SENTENCE")
    
    b1, b2, b3 = st.columns(3)
    if b1.button("Space ␣"): st.session_state.sentence += " "
    if b2.button("⌫"): st.session_state.sentence = st.session_state.sentence[:-1]
    if b3.button("🗑️", type="primary"): st.session_state.sentence = ""
    
    sentence_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# Update UI from Shared State
if webrtc_ctx.state.playing:
    with shared_state.lock:
        gesture_placeholder.markdown(f'<div class="prediction-panel"><h4>Gesture</h4><h2>{shared_state.gesture}</h2></div>', unsafe_allow_html=True)
        confidence_placeholder.markdown(f'<div class="prediction-panel"><h4>Confidence</h4><h2>{shared_state.confidence:.1%}</h2></div>', unsafe_allow_html=True)
    sentence_placeholder.markdown(f'<div class="sentence-box">{st.session_state.sentence}</div>', unsafe_allow_html=True)
    st.rerun() # Refresh to update session state changes from processor
else:
    st.info("Start the WebRTC stream to begin detection.")
