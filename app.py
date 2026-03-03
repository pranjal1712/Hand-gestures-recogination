import streamlit as st
import cv2
import numpy as np
import time
import threading
import os

# --- Core AI Imports ---
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    st.error(f"Environmental Error: Mediapipe failed to load. {e}")
    st.info("Technical Tip: This usually happens due to Protobuf version conflict on Hugging Face. Applying fix...")
    st.stop()

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
except ImportError:
    st.error("Dependency Error: 'streamlit-webrtc' not found. Please check requirements.txt.")
    st.stop()

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        st.error("Critical: TFLite Engine missing (TensorFlow or tflite-runtime).")
        st.stop()

# --- Configuration ---
MODEL_PATH = "model/asl_landmark_model.tflite"
LABEL_PATH = "model/label_map.npy"
STABILITY_THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.8
COOLDOWN_TIME = 1.2

# --- App Settings ---
st.set_page_config(
    page_title="Gesture Flow AI",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Check Assets ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    st.error(f"Critical Error: Required assets (model/labels) are missing in the repository.")
    st.stop()

# --- Style ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp { background-color: #0d1117; color: #f0f0f0; font-family: 'Inter', sans-serif; }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(1.8rem, 5vw, 3.2rem);
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        letter-spacing: 2px;
    }
    
    .sentence-display {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 12px;
        padding: 24px;
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(1.4rem, 4vw, 2.2rem);
        color: #00ffcc;
        text-align: center;
        min-height: 80px;
        border: 1px solid #00ffcc44;
        box-shadow: inset 0 0 15px rgba(0, 255, 204, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 10px;
    }
    
    .stat-label { color: #8899ac; font-size: 0.9rem; margin-bottom: 4px; }
    .stat-value { font-family: 'Orbitron', sans-serif; font-size: 1.4rem; color: #00d2ff; }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

class InferenceState:
    def __init__(self):
        self.gesture = "NONE"
        self.confidence = 0.0
        self.last_char = None
        self.stable_count = 0
        self.last_ts = 0
        self.lock = threading.Lock()

if 'inference_state' not in st.session_state:
    st.session_state.inference_state = InferenceState()

# --- Model Engine ---
@st.cache_resource
def init_engine():
    # Diagnostic Check
    if not os.path.exists(MODEL_PATH):
        st.error(f"MODEL NOT FOUND at {os.path.abspath(MODEL_PATH)}")
        st.stop()
    
    m_size = os.path.getsize(MODEL_PATH)
    if m_size < 1000:
        st.error(f"CORRUPT MODEL: Size {m_size} bytes (Possible Git LFS Pointer)")
        st.stop()

    try:
        interpreter = Interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
    except Exception as e:
        st.error(f"Engine Failure: {str(e)}")
        st.info("Attempting fallback engine...")
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(MODEL_PATH)
            interpreter.allocate_tensors()
        except Exception as e2:
            st.error(f"Critical Engine Crash: {e2}")
            st.write(f"Model Path: {os.path.abspath(MODEL_PATH)}")
            st.write(f"Debug Info: {m_size} bytes")
            st.stop()

    return {
        'itp': interpreter,
        'inp': interpreter.get_input_details()[0],
        'out': interpreter.get_output_details()[0],
        'lbl': np.load(LABEL_PATH, allow_pickle=True)
    }

engine = init_engine()

# --- Frame Processor ---
class HandProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        
        pred_gesture = "NONE"
        pred_conf = 0.0
        
        if result.multi_hand_landmarks:
            h_lms = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, h_lms, mp_hands.HAND_CONNECTIONS)
            
            # Predict
            vec = [coord for lm in h_lms.landmark for coord in (lm.x, lm.y, lm.z)]
            inp_data = np.array(vec, dtype=np.float32).reshape(1, -1)
            
            engine['itp'].set_tensor(engine['inp']["index"], inp_data)
            engine['itp'].invoke()
            probs = engine['itp'].get_tensor(engine['out']["index"])[0]
            idx = np.argmax(probs)
            pred_gesture = engine['lbl'][idx]
            pred_conf = float(probs[idx])

        # Push to Shared State
        is_state = st.session_state.inference_state
        with is_state.lock:
            is_state.gesture = pred_gesture
            is_state.confidence = pred_conf
            
            if pred_conf > CONFIDENCE_THRESHOLD:
                if pred_gesture == is_state.last_char:
                    is_state.stable_count += 1
                else:
                    is_state.last_char = pred_gesture
                    is_state.stable_count = 1
                
                if is_state.stable_count >= STABILITY_THRESHOLD:
                    now = time.time()
                    if now - is_state.last_ts > COOLDOWN_TIME:
                        char = pred_gesture.upper() if pred_gesture.upper() != "SPACE" else " "
                        st.session_state.sentence += char
                        is_state.last_ts = now
                        is_state.stable_count = 0
            else:
                is_state.stable_count = 0

        return frame.from_ndarray(img, format="bgr24")

# --- Layout ---
st.markdown('<h1 class="hero-title">GESTURE FLOW AI</h1>', unsafe_allow_html=True)

v_col, i_col = st.columns([1.6, 1])

with v_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    webrtc_streamer(
        key="asl-main",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HandProcessor,
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 20}
            },
            "audio": False
        },
        video_html_attrs={
            "style": {"width": "100%", "margin": "0 auto", "border-radius": "10px"},
            "controls": False,
            "autoPlay": True,
        },
    )
    st.markdown('</div>', unsafe_allow_html=True)

with i_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📡 Live Analytics")
    is_state = st.session_state.inference_state
    
    st.markdown(f'<div class="stat-label">DETECTED GESTURE</div><div class="stat-value">{is_state.gesture}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat-label">CONFIDENCE SCORE</div><div class="stat-value">{is_state.confidence:.1%}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⌨️ Composition")
    
    ctrl1, ctrl2 = st.columns(2)
    if ctrl1.button("Delete char ⌫", use_container_width=True):
        st.session_state.sentence = st.session_state.sentence[:-1]
    if ctrl2.button("Reset All 🗑️", type="primary", use_container_width=True):
        st.session_state.sentence = ""
        
    st.markdown(f'<div class="sentence-display">{st.session_state.sentence}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Loop to sync state changes from thread
if st.session_state.get("asl-main_playing", False):
    time.sleep(0.1)
    st.rerun()
