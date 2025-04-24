from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import tempfile
import os
import base64
import json
from datetime import datetime
from dotenv import load_dotenv
import time
import threading
import atexit
import math

# Load environment variables
load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize MediaPipe with environment variables
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity=1,
    static_image_mode=False,
    max_num_hands=1,               # Focus on one hand
    min_detection_confidence=0.9,  # Increased confidence threshold
    min_tracking_confidence=0.9    # Increased tracking threshold
)

# Gesture mappings
GESTURES = {
    'THUMBS_UP': '👍 Great!',
    'PEACE': '✌️ Peace',
    'OPEN_PALM': '🖐️ Hello',
    'FIST': '✊ Strong',
    'POINTING': '👆 Look',
}

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    angle = math.degrees(math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - 
                        math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    return abs(angle)

def detect_gesture(landmarks):
    """Highly accurate gesture detection using relative positions and angles."""
    if len(landmarks) != 21:
        return None

    # Extract all important landmarks
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    thumb_mcp = landmarks[2]
    thumb_ip = landmarks[3]
    thumb_tip = landmarks[4]
    
    index_mcp = landmarks[5]
    index_pip = landmarks[6]
    index_dip = landmarks[7]
    index_tip = landmarks[8]
    
    middle_mcp = landmarks[9]
    middle_pip = landmarks[10]
    middle_dip = landmarks[11]
    middle_tip = landmarks[12]
    
    ring_mcp = landmarks[13]
    ring_pip = landmarks[14]
    ring_dip = landmarks[15]
    ring_tip = landmarks[16]
    
    pinky_mcp = landmarks[17]
    pinky_pip = landmarks[18]
    pinky_dip = landmarks[19]
    pinky_tip = landmarks[20]

    # Calculate hand size for normalization
    hand_size = calculate_distance(wrist, middle_mcp)
    
    def is_finger_extended(tip, dip, pip, mcp):
        """Check if a finger is extended using angles and distances."""
        angle1 = calculate_angle(tip, dip, pip)
        angle2 = calculate_angle(dip, pip, mcp)
        dist = calculate_distance(tip, pip) / hand_size
        return angle1 > 160 and angle2 > 160 and dist > 0.15
    
    def is_finger_folded(tip, dip, pip, mcp):
        """Check if a finger is folded using distances and angles."""
        dist = calculate_distance(tip, mcp) / hand_size
        angle1 = calculate_angle(tip, dip, pip)
        angle2 = calculate_angle(dip, pip, mcp)
        return dist < 0.12 and angle1 < 100 and angle2 < 100

    # Check thumb position (using angle relative to index finger)
    thumb_angle = calculate_angle(thumb_tip, thumb_ip, index_mcp)
    thumb_extended = thumb_angle > 30
    thumb_folded = calculate_distance(thumb_tip, index_mcp) / hand_size < 0.1
    
    # Check each finger state
    index_extended = is_finger_extended(index_tip, index_dip, index_pip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_dip, middle_pip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_dip, ring_pip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_dip, pinky_pip, pinky_mcp)
    
    index_folded = is_finger_folded(index_tip, index_dip, index_pip, index_mcp)
    middle_folded = is_finger_folded(middle_tip, middle_dip, middle_pip, middle_mcp)
    ring_folded = is_finger_folded(ring_tip, ring_dip, ring_pip, ring_mcp)
    pinky_folded = is_finger_folded(pinky_tip, pinky_dip, pinky_pip, pinky_mcp)

    # Thumbs Up
    if (thumb_extended and
        calculate_angle(thumb_tip, thumb_mcp, wrist) > 45 and
        index_folded and middle_folded and ring_folded and pinky_folded and
        thumb_tip[1] < wrist[1]):  # Thumb above wrist
        return 'THUMBS_UP'

    # Peace Sign - More precise checks for finger separation and angles
    if (index_extended and middle_extended and
        ring_folded and pinky_folded and
        not thumb_extended):
        # Check if fingers are properly separated
        finger_separation = calculate_distance(index_tip, middle_tip) / hand_size
        if 0.12 < finger_separation < 0.25:  # Adjusted range for better detection
            # Check if fingers are roughly parallel
            index_angle = calculate_angle(index_tip, index_mcp, wrist)
            middle_angle = calculate_angle(middle_tip, middle_mcp, wrist)
            if abs(index_angle - middle_angle) < 20:  # Fingers should be roughly parallel
                return 'PEACE'

    # Open Palm
    if (index_extended and middle_extended and ring_extended and pinky_extended and
        thumb_extended and
        calculate_angle(thumb_tip, thumb_mcp, index_mcp) > 45):  # Thumb spread
        # Check fingers are aligned
        finger_heights = [tip[1] for tip in [index_tip, middle_tip, ring_tip, pinky_tip]]
        if max(finger_heights) - min(finger_heights) < 0.1:
            return 'OPEN_PALM'

    # Fist - More strict checks for all fingers being tightly folded
    if (index_folded and middle_folded and ring_folded and pinky_folded and
        thumb_folded):
        # Additional check for tight fist formation
        palm_center = [(index_mcp[0] + pinky_mcp[0])/2, (index_mcp[1] + pinky_mcp[1])/2]
        max_finger_distance = max(
            calculate_distance(tip, palm_center) / hand_size
            for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
        )
        if max_finger_distance < 0.15:  # All fingers should be close to palm center
            return 'FIST'

    # Pointing - More precise checks for index finger extension
    if (index_extended and
        middle_folded and ring_folded and pinky_folded and
        not thumb_extended):
        # Check if index finger is pointing upward
        index_angle = calculate_angle(index_tip, index_mcp, wrist)
        if index_angle > 150:  # Index finger should be pointing upward
            # Additional check for finger alignment
            if abs(index_tip[0] - index_mcp[0]) < 0.1:  # Finger should be relatively straight
                return 'POINTING'

    return None

def process_frame(frame):
    """Process frame and detect hand gestures with improved visualization."""
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    # Draw landmarks with improved visibility
                    mp_drawing.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Convert landmarks to list format
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    
                    gesture = detect_gesture(landmarks)
                    
                    # Add gesture text to frame with improved visibility
                    if gesture:
                        text = GESTURES.get(gesture, "")
                        cv2.putText(frame_rgb, text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Draw confidence and debug information
                        cv2.putText(frame_rgb, f"Confidence: {hand_landmarks.landmark[0].visibility:.2f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw finger state information
                        if len(landmarks) == 21:  # Make sure we have all landmarks
                            hand_size = calculate_distance(landmarks[0], landmarks[9])  # wrist to middle_mcp
                            cv2.putText(frame_rgb, f"Hand size: {hand_size:.2f}", (10, 90),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Show angles for debugging
                            thumb_angle = calculate_angle(landmarks[4], landmarks[3], landmarks[5])
                            index_angle = calculate_angle(landmarks[8], landmarks[6], landmarks[5])
                            cv2.putText(frame_rgb, f"Thumb angle: {thumb_angle:.1f}", (10, 120),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(frame_rgb, f"Index angle: {index_angle:.1f}", (10, 150),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error processing landmarks: {str(e)}")
                    continue
        
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), gesture
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return frame, None

def text_to_speech(text, lang=None):
    """Convert text to speech and return audio file path."""
    if lang is None:
        lang = os.getenv('SPEECH_LANG', 'en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts = gTTS(text=text, lang=lang)
        tts.save(fp.name)
        return fp.name

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.last_frame = None
        self.frame_lock = threading.Lock()
    
    def start(self):
        """Start the camera with proper initialization."""
        if self.is_running:
            return True
            
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Error: Could not open camera")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            return True
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            self.cleanup()
            return False
    
    def stop(self):
        """Stop the camera and clean up resources."""
        self.is_running = False
        self.cleanup()
    
    def cleanup(self):
        """Clean up camera resources."""
        if self.camera is not None:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
    
    def read_frame(self):
        """Read a frame from the camera with error handling."""
        if not self.is_running or self.camera is None:
            return None
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                print("Error reading frame")
                return None
                
            with self.frame_lock:
                self.last_frame = frame
            return frame
        except Exception as e:
            print(f"Error reading frame: {str(e)}")
            return None
    
    def get_last_frame(self):
        """Get the last successfully read frame."""
        with self.frame_lock:
            return self.last_frame

# Create a global camera manager instance
camera_manager = CameraManager()

def gen_frames():
    """Generate frames from webcam with gesture detection."""
    if not camera_manager.start():
        print("Failed to start camera")
        return
        
    try:
        last_gesture = None
        last_gesture_time = datetime.now()
        gesture_stability_count = 0
        required_stability = 3
        
        while camera_manager.is_running:
            try:
                frame = camera_manager.read_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame, gesture = process_frame(frame)
                
                # Handle gesture detection with stability check
                if gesture:
                    if gesture == last_gesture:
                        gesture_stability_count += 1
                    else:
                        gesture_stability_count = 1
                        last_gesture = gesture
                    
                    if gesture_stability_count >= required_stability:
                        if (datetime.now() - last_gesture_time).seconds >= 2:
                            text = GESTURES.get(gesture, "")
                            socketio.emit('gesture_detected', {'gesture': gesture, 'text': text})
                            last_gesture_time = datetime.now()
                else:
                    gesture_stability_count = 0
                    last_gesture = None
                
                # Encode frame for streaming
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    print("Error encoding frame")
                    continue
                    
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            except Exception as e:
                print(f"Error in frame processing: {str(e)}")
                time.sleep(0.1)
                continue
                
    except Exception as e:
        print(f"Error in frame generation: {str(e)}")
    finally:
        camera_manager.stop()

@app.route('/video_feed')
def video_feed():
    """Video streaming route with error handling."""
    try:
        return Response(gen_frames(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Stream error: {str(e)}")
        return "Error streaming video", 500

@app.route('/text_to_speech', methods=['POST'])
def generate_speech():
    """Generate speech from text."""
    data = request.get_json()
    text = data.get('text', '')
    
    if text:
        audio_file = text_to_speech(text)
        with open(audio_file, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        os.unlink(audio_file)
        return jsonify({'audio': audio_data})
    
    return jsonify({'error': 'No text provided'}), 400

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera when the page is closed."""
    camera_manager.stop()
    return "Camera stopped"

@app.route('/start_camera')
def start_camera():
    """Start the camera when the page is opened."""
    if camera_manager.start():
        return "Camera started"
    return "Failed to start camera", 500

# Add cleanup on application exit
atexit.register(camera_manager.stop)

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', '1').lower() == 'true'
    socketio.run(app, host=host, port=port, debug=debug) 