"""
AI-Assisted Alternate Nostril Breathing (Nadi Shodhana) Guide
Hackathon Project - Real-time posture and breathing correction system
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import queue
import math
import pyaudio
import pyttsx3
from collections import deque
from scipy import signal

class NadiShodhanaGuide:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose, face, and hand detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_queue = queue.Queue()
        self.last_alert_time = {}
        self.alert_cooldown = 3.0  # Seconds between same alerts
        
        # Initialize audio processing for breathing detection
        self.audio_queue = queue.Queue()
        self.breathing_phase = "idle"  # idle, inhale, exhale
        self.breath_start_time = time.time()
        self.breath_durations = deque(maxlen=5)
        self.target_breath_duration = 4.5  # Target 4-5 seconds
        
        # Audio parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.audio_stream = None
        
        # Breathing visualization parameters
        self.circle_radius = 50
        self.circle_max_radius = 100
        self.circle_min_radius = 30
        self.breathing_animation_phase = 0
        
        # Eye tracking parameters
        self.eyes_closed_start = None
        self.eyes_open_duration = 0
        
        # Nostril tracking
        self.current_nostril = None  # 'left' or 'right'
        self.nostril_sequence = ['right', 'left', 'right', 'left']  # Standard sequence
        self.nostril_index = 0
        
        # Start TTS thread
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.audio_worker, daemon=True)
        
    def tts_worker(self):
        """Worker thread for text-to-speech to avoid blocking"""
        while True:
            try:
                text = self.tts_queue.get(timeout=0.1)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                continue
    
    def audio_worker(self):
        """Worker thread for audio processing"""
        try:
            self.audio_stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            noise_level = None
            amplitude_buffer = deque(maxlen=10)
            
            while True:
                try:
                    data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # Calculate RMS amplitude
                    amplitude = np.sqrt(np.mean(audio_data**2))
                    amplitude_buffer.append(amplitude)
                    
                    # Calibrate noise level
                    if noise_level is None and len(amplitude_buffer) == 10:
                        noise_level = np.mean(amplitude_buffer) * 1.5
                    
                    if noise_level:
                        smooth_amplitude = np.mean(amplitude_buffer)
                        
                        # Detect breathing phases
                        if smooth_amplitude > noise_level * 1.2:
                            if self.breathing_phase != "inhale":
                                self.breathing_phase = "inhale"
                                self.breath_start_time = time.time()
                        elif smooth_amplitude < noise_level * 0.8:
                            if self.breathing_phase == "inhale":
                                self.breathing_phase = "exhale"
                                breath_duration = time.time() - self.breath_start_time
                                self.breath_durations.append(breath_duration)
                        
                        # Put amplitude in queue for main thread
                        if not self.audio_queue.full():
                            self.audio_queue.put(smooth_amplitude)
                            
                except Exception as e:
                    print(f"Audio processing error: {e}")
                    continue
                    
        except Exception as e:
            print(f"Audio stream error: {e}")
    
    def trigger_alert(self, alert_type, message):
        """Trigger audio alert with cooldown"""
        current_time = time.time()
        if alert_type not in self.last_alert_time or \
           current_time - self.last_alert_time[alert_type] > self.alert_cooldown:
            self.tts_queue.put(message)
            self.last_alert_time[alert_type] = current_time
            return True
        return False
    
    def check_spinal_alignment(self, pose_landmarks):
        """Check if spine is straight using ear-shoulder-hip alignment"""
        if not pose_landmarks:
            return True, 0
        
        # Get relevant landmarks
        left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Calculate midpoints
        ear_mid_x = (left_ear.x + right_ear.x) / 2
        ear_mid_y = (left_ear.y + right_ear.y) / 2
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_mid_x = (left_hip.x + right_hip.x) / 2
        hip_mid_y = (left_hip.y + right_hip.y) / 2
        
        # Calculate angle from vertical
        dx1 = shoulder_mid_x - ear_mid_x
        dy1 = shoulder_mid_y - ear_mid_y
        dx2 = hip_mid_x - shoulder_mid_x
        dy2 = hip_mid_y - shoulder_mid_y
        
        # Calculate angles from vertical (90 degrees is perfect vertical)
        angle1 = math.degrees(math.atan2(dx1, dy1))
        angle2 = math.degrees(math.atan2(dx2, dy2))
        
        # Average deviation from vertical
        avg_deviation = (abs(angle1) + abs(angle2)) / 2
        
        # Check if within acceptable range (10-15 degrees)
        is_aligned = avg_deviation < 15
        
        if not is_aligned:
            self.trigger_alert("spine", "Straighten your back")
        
        return is_aligned, avg_deviation
    
    def check_head_position(self, face_landmarks):
        """Check head/neck position"""
        if not face_landmarks:
            return True, "centered"
        
        # Get nose tip and ear landmarks
        nose_tip = face_landmarks.landmark[1]  # Nose tip
        left_ear = face_landmarks.landmark[234]  # Left ear tragion
        right_ear = face_landmarks.landmark[454]  # Right ear tragion
        
        # Check for forward tilt (nose too far from ears)
        ear_mid_x = (left_ear.x + right_ear.x) / 2
        ear_mid_y = (left_ear.y + right_ear.y) / 2
        
        nose_ear_dist = math.sqrt((nose_tip.x - ear_mid_x)**2 + (nose_tip.y - ear_mid_y)**2)
        
        # Check for lateral tilt
        ear_level_diff = abs(left_ear.y - right_ear.y)
        
        head_status = "centered"
        if nose_ear_dist > 0.15:  # Forward tilt threshold
            head_status = "forward"
            self.trigger_alert("head", "Keep head centered")
        elif ear_level_diff > 0.05:  # Lateral tilt threshold
            head_status = "tilted"
            self.trigger_alert("head", "Keep head centered")
        
        return head_status == "centered", head_status
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio for eye openness detection"""
        # Eye landmarks indices for MediaPipe
        # Left eye: 33, 160, 158, 133, 153, 144
        # Right eye: 362, 385, 387, 263, 373, 380
        
        def eye_distance(p1, p2):
            return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
        
        # Calculate for left eye
        left_vertical_1 = eye_distance(eye_landmarks[160], eye_landmarks[144])
        left_vertical_2 = eye_distance(eye_landmarks[158], eye_landmarks[153])
        left_horizontal = eye_distance(eye_landmarks[33], eye_landmarks[133])
        
        left_ear = (left_vertical_1 + left_vertical_2) / (2.0 * left_horizontal) if left_horizontal > 0 else 0
        
        # Calculate for right eye
        right_vertical_1 = eye_distance(eye_landmarks[385], eye_landmarks[380])
        right_vertical_2 = eye_distance(eye_landmarks[387], eye_landmarks[373])
        right_horizontal = eye_distance(eye_landmarks[362], eye_landmarks[263])
        
        right_ear = (right_vertical_1 + right_vertical_2) / (2.0 * right_horizontal) if right_horizontal > 0 else 0
        
        return (left_ear + right_ear) / 2
    
    def check_eyes_closed(self, face_landmarks):
        """Check if eyes are closed using EAR method"""
        if not face_landmarks:
            return True, 0
        
        ear = self.calculate_ear(face_landmarks.landmark)
        eyes_closed = ear < 0.15  # Threshold for closed eyes
        
        current_time = time.time()
        
        if not eyes_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            
            self.eyes_open_duration = current_time - self.eyes_closed_start
            
            if self.eyes_open_duration > 2.0:  # Eyes open for more than 2 seconds
                self.trigger_alert("eyes", "Close your eyes")
        else:
            self.eyes_closed_start = None
            self.eyes_open_duration = 0
        
        return eyes_closed, ear
    
    def detect_nostril_closure(self, hand_landmarks, face_landmarks):
        """Detect which nostril is being closed by the hand"""
        if not hand_landmarks or not face_landmarks:
            return None
        
        # Get thumb and index finger tips
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Get nose tip position
        nose_tip = face_landmarks.landmark[1]
        
        # Calculate distances to nose
        thumb_to_nose = math.sqrt((thumb_tip.x - nose_tip.x)**2 + (thumb_tip.y - nose_tip.y)**2)
        index_to_nose = math.sqrt((index_tip.x - nose_tip.x)**2 + (index_tip.y - nose_tip.y)**2)
        
        # Determine which nostril based on finger position
        if thumb_to_nose < 0.1 or index_to_nose < 0.1:
            # Check if finger is on left or right side of nose
            finger_x = thumb_tip.x if thumb_to_nose < index_to_nose else index_tip.x
            
            if finger_x < nose_tip.x:
                return "right"  # Finger on left side closes right nostril
            else:
                return "left"   # Finger on right side closes left nostril
        
        return None
    
    def check_breathing_rhythm(self):
        """Check breathing rhythm and provide feedback"""
        if len(self.breath_durations) < 2:
            return True, "calibrating"
        
        avg_duration = np.mean(self.breath_durations)
        
        feedback = "good"
        if avg_duration < 3.5:
            feedback = "too_fast"
            self.trigger_alert("breathing", "Slow down your breathing")
        elif avg_duration > 5.5:
            feedback = "too_slow"
            self.trigger_alert("breathing", "Speed up slightly")
        
        return feedback == "good", feedback
    
    def draw_breathing_guide(self, img):
        """Draw visual breathing guide circle"""
        h, w = img.shape[:2]
        center = (w - 100, 100)
        
        # Animate circle based on breathing phase
        if self.breathing_phase == "inhale":
            self.circle_radius = min(self.circle_radius + 2, self.circle_max_radius)
            color = (0, 255, 0)  # Green for inhale
        elif self.breathing_phase == "exhale":
            self.circle_radius = max(self.circle_radius - 2, self.circle_min_radius)
            color = (0, 0, 255)  # Red for exhale
        else:
            color = (255, 255, 255)  # White for idle
        
        # Draw circle
        cv2.circle(img, center, self.circle_radius, color, 2)
        cv2.putText(img, self.breathing_phase.upper(), (center[0] - 40, center[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw target breathing indicator
        cv2.putText(img, "Target: 4-5 sec", (w - 180, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_status_panel(self, img, statuses):
        """Draw status panel with all checks"""
        h, w = img.shape[:2]
        panel_x = 10
        panel_y = 10
        line_height = 30
        
        # Background for better readability
        cv2.rectangle(img, (panel_x - 5, panel_y - 5), 
                     (panel_x + 300, panel_y + line_height * 6), 
                     (0, 0, 0), -1)
        cv2.rectangle(img, (panel_x - 5, panel_y - 5), 
                     (panel_x + 300, panel_y + line_height * 6), 
                     (255, 255, 255), 2)
        
        # Draw status items
        items = [
            ("Spine Alignment", statuses['spine_aligned'], 
             f"({statuses['spine_deviation']:.1f}°)"),
            ("Head Position", statuses['head_centered'], 
             f"({statuses['head_status']})"),
            ("Eyes Closed", statuses['eyes_closed'], 
             f"(EAR: {statuses['ear']:.2f})"),
            ("Nostril", statuses['nostril_correct'], 
             f"({statuses['current_nostril'] or 'none'})"),
            ("Breathing Rhythm", statuses['breathing_good'], 
             f"({statuses['breathing_feedback']})")
        ]
        
        for i, (label, status, info) in enumerate(items):
            y = panel_y + (i + 1) * line_height
            color = (0, 255, 0) if status else (0, 0, 255)
            symbol = "✓" if status else "✗"
            
            cv2.putText(img, f"{symbol} {label}: {info}", (panel_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def run(self):
        """Main loop for the application"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Start audio thread
        self.audio_thread.start()
        
        print("Nadi Shodhana Guide Started!")
        print("Press 'q' to quit")
        print("Press 's' to start/reset breathing sequence")
        
        # Initial instruction
        self.tts_queue.put("Welcome to Nadi Shodhana practice. Sit comfortably with your spine straight.")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
            
            # Flip image horizontally for selfie view
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            pose_results = self.pose.process(image_rgb)
            face_results = self.face_mesh.process(image_rgb)
            hands_results = self.hands.process(image_rgb)
            
            # Status tracking
            statuses = {
                'spine_aligned': True,
                'spine_deviation': 0,
                'head_centered': True,
                'head_status': 'centered',
                'eyes_closed': True,
                'ear': 0,
                'nostril_correct': True,
                'current_nostril': None,
                'breathing_good': True,
                'breathing_feedback': 'good'
            }
            
            # Check spinal alignment
            if pose_results.pose_landmarks:
                aligned, deviation = self.check_spinal_alignment(pose_results.pose_landmarks)
                statuses['spine_aligned'] = aligned
                statuses['spine_deviation'] = deviation
                
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            
            # Check head position and eyes
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # Head position
                centered, head_status = self.check_head_position(face_landmarks)
                statuses['head_centered'] = centered
                statuses['head_status'] = head_status
                
                # Eyes closed
                eyes_closed, ear = self.check_eyes_closed(face_landmarks)
                statuses['eyes_closed'] = eyes_closed
                statuses['ear'] = ear
            
            # Check nostril and hand position
            if hands_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                hand_landmarks = hands_results.multi_hand_landmarks[0]
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # Detect which nostril is closed
                nostril = self.detect_nostril_closure(hand_landmarks, face_landmarks)
                statuses['current_nostril'] = nostril
                
                # TODO: Advanced nostril breath detection via microphone
                # This would involve spectral analysis of breath sounds
                # to determine which nostril air is flowing through
                # For hackathon demo, we rely on hand position detection
                
                # Check if correct nostril for sequence
                if nostril and self.nostril_index < len(self.nostril_sequence):
                    expected_nostril = self.nostril_sequence[self.nostril_index]
                    if nostril != expected_nostril:
                        statuses['nostril_correct'] = False
                        self.trigger_alert("nostril", f"Use {expected_nostril} nostril")
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )
            
            # Check breathing rhythm
            rhythm_good, feedback = self.check_breathing_rhythm()
            statuses['breathing_good'] = rhythm_good
            statuses['breathing_feedback'] = feedback
            
            # Draw UI elements
            self.draw_breathing_guide(image)
            self.draw_status_panel(image, statuses)
            
            # Show frame
            cv2.imshow('Nadi Shodhana Guide', image)
            
            # Handle keyboard input
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.nostril_index = 0
                self.tts_queue.put("Starting breathing sequence. Close right nostril first.")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        self.p.terminate()

def main():
    """Main entry point"""
    print("=" * 60)
    print("AI-Assisted Nadi Shodhana (Alternate Nostril Breathing)")
    print("=" * 60)
    print("\nInitializing... Please wait...")
    
    try:
        guide = NadiShodhanaGuide()
        print("\nInitialization complete!")
        print("\nInstructions:")
        print("- Sit in a comfortable position with your back straight")
        print("- Keep your eyes closed during practice")
        print("- Follow the breathing circle: Green = Inhale, Red = Exhale")
        print("- Use your thumb and ring finger to alternately close nostrils")
        print("- Press 'S' to start the guided sequence")
        print("- Press 'Q' to quit\n")
        
        guide.run()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a webcam connected")
        print("2. Install required packages:")
        print("   pip install opencv-python mediapipe numpy scipy pyaudio pyttsx3")
        print("3. On Linux/Mac, you might need to install PortAudio:")
        print("   Linux: sudo apt-get install portaudio19-dev")
        print("   Mac: brew install portaudio")

if __name__ == "__main__":
    main()