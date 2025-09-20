"""
Flask Web Application for AI-Assisted Nadi Shodhana (Alternate Nostril Breathing)
"""

from flask import Flask, render_template, Response, request, jsonify
import threading
import time
import numpy as np
from breathing_guide import NadiShodhanaGuide

app = Flask(__name__)

# Global instance of the breathing guide
breathing_guide = None

def init_breathing_guide():
    """Initialize the breathing guide in a separate thread"""
    global breathing_guide
    if breathing_guide is None:
        breathing_guide = NadiShodhanaGuide()
        print("Breathing guide initialized")

@app.route('/')
def home():
    """Home page with introduction and navigation"""
    return render_template('home.html')

@app.route('/guide')
def guide():
    """Breathing guide page with camera controls"""
    return render_template('guide.html')

@app.route('/start_camera')
def start_camera():
    """Start the camera feed"""
    global breathing_guide
    if breathing_guide is None:
        init_breathing_guide()
    
    breathing_guide.start_camera()
    return jsonify({'status': 'success', 'message': 'Camera started'})

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera feed"""
    global breathing_guide
    if breathing_guide:
        breathing_guide.stop_camera()
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/start_breathing')
def start_breathing():
    """Start the breathing sequence"""
    global breathing_guide
    if breathing_guide:
        breathing_guide.start_breathing_sequence()
    return jsonify({'status': 'success', 'message': 'Breathing sequence started'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global breathing_guide
    if breathing_guide is None:
        init_breathing_guide()
    
    if not breathing_guide.camera_active:
        breathing_guide.start_camera()
    
    return Response(breathing_guide.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """Get current breathing guide status with enhanced tracking data"""
    global breathing_guide
    if breathing_guide is None:
        return jsonify({'status': 'not_initialized'})
    
    # Calculate rhythm score based on breathing durations
    rhythm_score = 50  # Default score
    rhythm_feedback = "Calibrating..."
    
    if len(breathing_guide.breath_durations) >= 2:
        avg_duration = np.mean(breathing_guide.breath_durations)
        target_duration = breathing_guide.target_breath_duration
        
        # Calculate score based on how close to healthy 4-6 second range
        if 4.0 <= avg_duration <= 6.0:
            # Perfect range
            rhythm_score = 95
            rhythm_feedback = "Perfect rhythm! (4-6s)"
        elif 3.5 <= avg_duration <= 6.5:
            # Good range
            rhythm_score = 85
            rhythm_feedback = "Good rhythm"
        elif 3.0 <= avg_duration <= 7.0:
            # Acceptable range
            rhythm_score = 70
            rhythm_feedback = "Fair rhythm"
        elif avg_duration < 3.0:
            # Too fast - unhealthy
            rhythm_score = 30
            rhythm_feedback = "Too fast - slow down"
        else:
            # Too slow
            rhythm_score = 40
            rhythm_feedback = "Too slow - speed up"
    
    return jsonify({
        'status': 'active',
        'camera_active': breathing_guide.camera_active,
        'breathing_phase': breathing_guide.breathing_phase,
        'current_nostril': breathing_guide.current_nostril,
        'nostril_index': breathing_guide.nostril_index,
        'spine_status': 'Good',  # This would come from actual detection
        'eyes_status': 'Closed',  # This would come from actual detection
        'head_status': 'Centered',  # This would come from actual detection
        'rhythm_score': rhythm_score,
        'rhythm_feedback': rhythm_feedback,
        'session_duration': time.time() - getattr(breathing_guide, 'session_start_time', time.time()),
        'breath_durations': list(breathing_guide.breath_durations),
        'target_duration': breathing_guide.target_breath_duration,
        'cycle_progress': getattr(breathing_guide, 'cycle_progress', 0),
        'circle_scale': getattr(breathing_guide, 'circle_scale', 0)
    })

@app.route('/toggle_audio', methods=['POST'])
def toggle_audio():
    """Toggle audio detection on/off"""
    global breathing_guide
    if breathing_guide:
        data = request.get_json()
        enabled = data.get('enabled', True)
        breathing_guide.audio_enabled = enabled
        return jsonify({'status': 'success', 'audio_enabled': enabled})
    return jsonify({'status': 'error', 'message': 'Breathing guide not initialized'})

@app.route('/cleanup')
def cleanup():
    """Cleanup resources"""
    global breathing_guide
    if breathing_guide:
        breathing_guide.cleanup()
        breathing_guide = None
    return jsonify({'status': 'success', 'message': 'Resources cleaned up'})

if __name__ == '__main__':
    # Initialize breathing guide in background
    init_thread = threading.Thread(target=init_breathing_guide, daemon=True)
    init_thread.start()
    
    print("Starting InhaleAI Flask application...")
    print("Visit http://localhost:5000 to access the application")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
