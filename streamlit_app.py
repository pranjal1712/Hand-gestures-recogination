import streamlit as st
import cv2
import numpy as np
import time
import threading
import os

import sys
import os

# --- System Debug Info ---
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = f"Python: {sys.version} | OS: {sys.platform}"

# --- Robust Imports ---
try:
    import mediapipe as mp
    from mediapipe.solutions import hands as mp_hands
    from mediapipe.solutions import drawing_utils as mp_drawing
    
    # Verify solutions are actually there
    _ = mp_hands.Hands
    _ = mp_drawing.draw_landmarks
except Exception as e:
    st.error(f"Critical: Mediapipe Setup Failed. {st.session_state.debug_info}")
    st.error(f"Error Details: {e}")
    st.info("Tip: Try to 'Reboot App' from the Streamlit Cloud menu to clear the cache.")
    st.stop()

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
except ImportError:
    st.error("Critical: 'streamlit-webrtc' is missing. Please check requirements.txt.")
    st.stop()

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        st.error("Critical: TensorFlow/TFLite Interpreter not found.")
        st.stop()

# --- Configuration ---
MODEL_PATH = "model/asl_landmark_model.tflite"
LABEL_PATH = "model/label_map.npy"
STABILITY_THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.8
COOLDOWN_TIME = 1.0

# --- Page Config ---
st.set_page_config(
    page_title="ASL Gesture Flow AI",
    page_icon="🖐️",
    layout="wide"
)

# --- Check Files ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    st.error(f"Critical: Model files missing at {MODEL_PATH} or {LABEL_PATH}")
    st.stop()

# --- Session State ---
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .glass-container { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-radius: 20px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px; }
    .main-title { font-family: 'Orbitron', sans-serif; font-size: clamp(2rem, 5vw, 3rem); background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1.5rem; font-weight: 700; }
    .sentence-box { background: rgba(0, 0, 0, 0.4); border: 1px solid #4CAF50; padding: 20px; border-radius: 15px; min-height: 80px; font-size: 1.8rem; color: #92FE9D; font-family: 'Orbitron', sans-serif; text-align: center; margin-top: 15px; word-wrap: break-word; }
    .status-panel { text-align: center; padding: 15px; background: rgba(255, 255, 255, 0.03); border-radius: 12px; border-left: 4px solid #00C9FF; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Global Shared State ---
class GlobalState:
    def __init__(self):
        self.gesture = "NONE"
        self.confidence = 0.0
        self.last_char = None
        self.stable_count = 0
        self.last_ts = 0
        self.lock = threading.Lock()

if 'global_state' not in st.session_state:
    st.session_state.global_state = GlobalState()

# --- Load Model ---
@st.cache_resource
def load_engine():
    try:
        interpreter = Interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        labels = np.load(LABEL_PATH, allow_pickle=True)
        return interpreter, input_details, output_details, labels
    except Exception as e:
        st.error(f"Engine Load Failed: {e}")
        return None

engine = load_engine()

# --- Processor Class ---
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        best_gesture = "NONE"
        best_conf = 0.0
        
        if result.multi_hand_landmarks and engine:
            interpreter, input_details, output_details, labels = engine
            for hand_lms in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # Extract and Predict
                vec = []
                for lm in hand_lms.landmark:
                    vec.extend([lm.x, lm.y, lm.z])
                inp = np.array(vec, dtype=np.float32).reshape(1, -1)
                
                interpreter.set_tensor(input_details["index"], inp)
                interpreter.invoke()
                probs = interpreter.get_tensor(output_details["index"])[0]
                idx = np.argmax(probs)
                best_gesture = labels[idx]
                best_conf = float(probs[idx])

        # Write to Shared State
        gs = st.session_state.global_state
        with gs.lock:
            gs.gesture = best_gesture
            gs.confidence = best_conf
            
            # Sentence Logic
            if best_conf > CONFIDENCE_THRESHOLD:
                if best_gesture == gs.last_char:
                    gs.stable_count += 1
                else:
                    gs.last_char = best_gesture
                    gs.stable_count = 1
                
                if gs.stable_count >= STABILITY_THRESHOLD:
                    now = time.time()
                    if now - gs.last_ts > COOLDOWN_TIME:
                        char = best_gesture.upper() if best_gesture.upper() != "SPACE" else " "
                        st.session_state.sentence += char
                        gs.last_ts = now
                        gs.stable_count = 0
            else:
                gs.stable_count = 0

        return frame.from_ndarray(img, format="bgr24")

# --- UI Layout ---
st.markdown('<div class="main-title">🖐️ ASL GESTURE FLOW</div>', unsafe_allow_html=True)
col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="asl-flow",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=ASLProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 🔍 Live Inference")
    gs = st.session_state.global_state
    
    st.markdown(f'<div class="status-panel">Gesture: <b>{gs.gesture}</b></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-panel">Confidence: <b>{gs.confidence:.1%}</b></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📝 Output")
    
    b_col1, b_col2 = st.columns(2)
    if b_col1.button("Backspace ⌫", use_container_width=True):
        st.session_state.sentence = st.session_state.sentence[:-1]
    if b_col2.button("Clear 🗑️", type="primary", use_container_width=True):
        st.session_state.sentence = ""
        
    st.markdown(f'<div class="sentence-box">{st.session_state.sentence}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if ctx.state.playing:
    time.sleep(0.1)
    st.rerun()
