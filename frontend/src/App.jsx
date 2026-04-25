import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import './index.css';

const Hands = window.Hands;
if (!Hands) console.error("MediaPipe Hands not found on window");
if (!window.drawConnectors) console.error("MediaPipe Drawing Utils not found on window");

const API_URL = '/predict';
const STABILITY_THRESHOLD = 3;
const COOLDOWN_TIME = 800; // ms
const LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'YY', 'Z'];

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState('Wait...');
  const [confidence, setConfidence] = useState(0);
  const [sentence, setSentence] = useState('');
  const [loading, setLoading] = useState(true);

  // Stability tracking
  const stabilityRef = useRef({
    lastLabel: '',
    count: 0,
    lastTime: 0
  });

  useEffect(() => {
    if (!Hands) return;

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7
    });

    hands.onResults(onResults);

    if (videoRef.current) {
      const videoElement = videoRef.current;
      const camera = new window.Camera(videoElement, {
        onFrame: async () => {
          await hands.send({ image: videoElement });
        },
        width: 640,
        height: 480
      });
      camera.start().then(() => setLoading(false));
    }

    return () => {
      hands.close();
    };
  }, [loading]);

  const onResults = async (results) => {
    const canvasCtx = canvasRef.current.getContext('2d');
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    // Mirror the entire canvas
    canvasCtx.translate(canvasRef.current.width, 0);
    canvasCtx.scale(-1, 1);

    // 2. Draw Live Hand
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0];

      window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, {
        color: '#00ffcc',
        lineWidth: 4,
      });
      window.drawLandmarks(canvasCtx, landmarks, {
        color: '#ffffff',
        lineWidth: 1,
        radius: 3,
      });

      // Prepare data for prediction
      const flatLandmarks = [];
      landmarks.forEach(lm => {
        flatLandmarks.push(1 - lm.x); // Un-mirroring for the model
        flatLandmarks.push(lm.y);
        flatLandmarks.push(lm.z);
      });

      try {
        const response = await axios.post(API_URL, { landmarks: flatLandmarks });
        const { label, confidence } = response.data;
        setPrediction(label);
        setConfidence(confidence);
        handleStability(label);
      } catch (err) {
        console.error("Prediction error:", err);
      }
    } else {
      setPrediction('None');
      setConfidence(0);
    }
    canvasCtx.restore();
  };

  const handleStability = (label) => {
    const now = Date.now();
    const s = stabilityRef.current;

    if (label === s.lastLabel) {
      s.count += 1;
    } else {
      s.lastLabel = label;
      s.count = 1;
    }

    if (s.count >= STABILITY_THRESHOLD && now - s.lastTime > COOLDOWN_TIME) {
      if (label === 'SPACE') {
        setSentence(prev => prev + ' ');
      } else {
        setSentence(prev => prev + label);
      }
      s.lastTime = now;
      s.count = 0; // reset
    }
  };

  return (
    <div className="dashboard">
      <header className="main-header">
        <h1>GESTURE FLOW <span>AI</span></h1>
        <p className="subtitle">High-Precision Sign Language Intelligence</p>
      </header>

      <div className="layout-grid">
        {/* LEFT COLUMN: Camera Feed & Instructions */}
        <div className="camera-container">
          <div className="video-wrapper">
            <video ref={videoRef} className="webcam-view" autoPlay playsInline muted />
            <canvas ref={canvasRef} className="webcam-canvas" width="640" height="480" />
            {loading && <div className="overlay-loader">Initializing Neural Engine...</div>}
          </div>
          
          <div className="glass-card">
            <h2 className="panel-title">💡 Usage Guide</h2>
            <ul style={{ fontSize: '0.85rem', color: '#94a3b8', paddingLeft: '1.2rem' }}>
              <li>Position your hand clearly in the frame.</li>
              <li>Hold a gesture for 1s to "type" it.</li>
              <li>Use 'SPACE' gesture to add spaces between words.</li>
            </ul>
          </div>
        </div>

        {/* RIGHT COLUMN: Analytics & Composition */}
        <div className="side-panel">
          <div className="glass-card">
            <h2 className="panel-title">📡 Live Analytics</h2>
            <div className="stat-group">
              <div className="stat-label">Detected Gesture</div>
              <div className="neon-text">{prediction}</div>
            </div>
            <div className="stat-group">
              <div className="stat-label">System Confidence</div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ 
                    width: `${confidence * 100}%`, 
                    backgroundColor: confidence > 0.8 ? '#00ffcc' : '#ef4444' 
                  }}
                ></div>
              </div>
              <div className="stat-small">{(confidence * 100).toFixed(1)}% Recognition Accuracy</div>
            </div>
          </div>

          <div className="glass-card">
            <h2 className="panel-title">📝 Text Composition</h2>
            <div className="sentence-box">
              {sentence || <span className="placeholder">Gestured text will appear here...</span>}
            </div>
            <div className="button-group">
              <button onClick={() => setSentence('')} className="danger">Clear</button>
              <button 
                onClick={() => {
                  const speech = new SpeechSynthesisUtterance(sentence);
                  window.speechSynthesis.speak(speech);
                }}
                disabled={!sentence}
              >
                Speak
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <footer className="footer">
        Hand Gesture Neural Engine v2.0 • Powered by MediaPipe & TensorFlow
      </footer>
    </div>
  );
}

export default App;
