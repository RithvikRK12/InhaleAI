Inhale AI - AI-Assisted Nadi Shodhana (Alternate Nostril Breathing)
Overview
Inhale AI is an intelligent breathing guide that uses computer vision and audio processing to provide real-time feedback during Nadi Shodhana (alternate nostril breathing) practice.

Features
Core Functionality (As Specified)
Pre-Camera Audio Instructions: Delivers comprehensive Vishnu Mudra instructions before camera activation
Intelligent Camera Management: Activates camera only after audio message delivery
Real-time Posture Monitoring:
Spine posture parameter monitoring (>2 threshold triggers correction)
Head tilt detection with continuous monitoring
Audio Feedback System: Provides immediate voice guidance for posture corrections
Technical Implementation
Computer Vision: MediaPipe for pose, face, and hand detection
Audio Processing: Real-time breathing detection via microphone
Text-to-Speech: Automated voice guidance using pyttsx3
Web Interface: Flask-based web application for easy access
Usage
Web Application
python app.py
Then visit http://localhost:5000 in your browser.

Direct Testing
python test_inhale_ai.py
Command Line Interface
python hack.py
Requirements
Python 3.7+
Webcam
Audio input/output devices
Required packages (see requirements.txt)
Installation
pip install -r requirements.txt
How It Works
Step 1: Audio Message Delivery
Before starting the camera, Inhale AI delivers the following audio instructions:

"Sit comfortably with a straight spine, then adopt Vishnu Mudra with your right hand. Close your right nostril with your thumb, inhale through your left, close it with your ring finger, then open your right nostril and exhale. Inhale through the right nostril, close it with your thumb, and exhale slowly through the left nostril to complete one cycle. Repeat for several rounds, starting with 3-5 and gradually increasing to 10-15 rounds."

Step 2: Camera Activation
After delivering the complete audio message (8-second wait), the camera is activated to begin real-time monitoring.

Step 3: Continuous Monitoring
While the camera is running, Inhale AI continuously monitors:

Spine Posture Monitoring
Calculates spine posture parameter from pose landmarks
If parameter > 2: Outputs "Please keep your spine straight."
Uses ear-shoulder-hip alignment for accurate detection
Head Tilt Monitoring
Detects forward and lateral head tilts
If head is tilted: Outputs "Please do not tilt your head."
Uses nose-ear landmark analysis for precise detection
Nostril Detection
Detects which nostril is being closed by hand position
Displays "left" or "right" in the status panel
Uses finger-to-nose distance analysis for accurate detection
Session Timer
Displays current session duration in MM:SS format
Shows in the camera window for tracking practice time
Technical Details
Spine Posture Parameter
The spine posture parameter is calculated as:

spine_posture_parameter = avg_deviation / 10.0
Where avg_deviation is the average deviation from vertical alignment of ear-shoulder-hip landmarks.

Head Tilt Detection
Forward Tilt: Distance from nose tip to ear midpoint > 0.15
Lateral Tilt: Difference in ear heights > 0.05
Audio Alert System
Cooldown period of 3 seconds between identical alerts
Non-blocking TTS processing in separate thread
Queue-based audio message delivery
File Structure
InhaleAI/
├── app.py                 # Flask web application
├── breathing_guide.py     # Core AI functionality
├── hack.py               # Command-line interface
├── test_inhale_ai.py     # Test script
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
├── static/              # CSS and JavaScript
└── README.md            # This file
Troubleshooting
Camera not working: Ensure webcam is connected and not used by other applications
Audio issues: Check audio device permissions and TTS engine installation
Detection problems: Ensure good lighting and clear view of face/body
Performance issues: Close other applications using camera/microphone
Contributing
This is a hackathon project demonstrating AI-assisted breathing guidance. Feel free to extend and improve the functionality.

License
Open source - feel free to use and modify for educational purposes.
