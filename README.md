# Inhale AI – AI-Assisted Nadi Shodhana (Alternate Nostril Breathing)

## Overview

Inhale AI is an intelligent breathing guidance system that uses computer vision and audio processing to provide real-time feedback during the practice of Nadi Shodhana (alternate nostril breathing). It monitors posture, head alignment, and nostril usage while delivering corrective voice instructions.

---

## Features

### Core Functionality

* **Pre-Camera Audio Instructions**
  Provides complete Vishnu Mudra instructions before activating the camera.

* **Intelligent Camera Management**
  Activates the webcam only after the instructional audio is delivered.

* **Real-Time Posture Monitoring**

  * Spine posture monitoring (threshold-based correction)
  * Head tilt detection (forward and lateral)

* **Audio Feedback System**
  Delivers real-time voice feedback for posture and alignment corrections.

---

## Technical Implementation

* **Computer Vision**: MediaPipe (pose, face, and hand detection)
* **Audio Processing**: Real-time breathing detection via microphone
* **Text-to-Speech**: pyttsx3 for automated voice guidance
* **Web Interface**: Flask-based application

---

## Usage

### Web Application

```bash
python app.py
```

Open in browser:

```
http://localhost:5000
```

### Direct Testing

```bash
python test_inhale_ai.py
```

### Command Line Interface

```bash
python hack.py
```

---

## Requirements

* Python 3.7+
* Webcam
* Microphone and audio output devices
* Required Python packages (see `requirements.txt`)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How It Works

### Step 1: Audio Instruction Delivery

Before camera activation, the system delivers the following guidance:

"Sit comfortably with a straight spine, then adopt Vishnu Mudra with your right hand. Close your right nostril with your thumb, inhale through your left, close it with your ring finger, then open your right nostril and exhale. Inhale through the right nostril, close it with your thumb, and exhale slowly through the left nostril to complete one cycle. Repeat for several rounds, starting with 3–5 and gradually increasing to 10–15 rounds."

---

### Step 2: Camera Activation

After the instruction audio (approximately 8 seconds), the webcam is activated.

---

### Step 3: Continuous Monitoring

#### Spine Posture Monitoring

* Uses ear–shoulder–hip alignment
* Computes posture parameter:

  ```
  spine_posture_parameter = avg_deviation / 10.0
  ```
* If parameter > 2:

  ```
  "Please keep your spine straight."
  ```

#### Head Tilt Detection

* Forward tilt:

  * Nose-to-ear midpoint distance > 0.15
* Lateral tilt:

  * Difference in ear height > 0.05
* Audio feedback:

  ```
  "Please do not tilt your head."
  ```

#### Nostril Detection

* Detects which nostril is closed based on finger-to-nose distance
* Displays "left" or "right" in the interface

#### Session Timer

* Displays elapsed time in MM:SS format
* Visible in camera interface

---

## Audio Alert System

* Cooldown period: 3 seconds between repeated alerts
* Non-blocking TTS using separate thread
* Queue-based audio message handling

---

## Project Structure

```
InhaleAI/
├── app.py                 # Flask web application
├── breathing_guide.py     # Core AI functionality
├── hack.py                # Command-line interface
├── test_inhale_ai.py      # Test script
├── requirements.txt       # Dependencies
├── templates/             # HTML templates
├── static/                # CSS and JavaScript
└── README.md              # Documentation
```

---

## Troubleshooting

* **Camera not working**
  Ensure webcam is connected and not used by another application

* **Audio issues**
  Check microphone permissions and TTS engine setup

* **Detection inaccuracies**
  Ensure proper lighting and clear visibility of face and upper body

* **Performance issues**
  Close other applications using camera or microphone

---

## Contributing

This project was developed as part of a hackathon. Contributions and improvements are welcome.

---

## License

Open source. Free to use and modify for educational purposes.
