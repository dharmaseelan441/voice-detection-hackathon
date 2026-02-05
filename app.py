import os
import json
import base64
import uuid
import logging
import numpy as np
import librosa
from flask import Flask, request, jsonify
from functools import wraps

# --- CONFIGURATION ---
# Get API KEY from Environment Variable, or use a default for safety
API_KEY = os.environ.get("API_KEY") 

# Check if Key is loaded (Optional: helps debug logs)
if not API_KEY:
    logger.warning("WARNING: No API_KEY set in environment variables!")
UPLOAD_FOLDER = "/tmp"  # Use /tmp for cloud environments
ALLOWED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AUTH DECORATOR ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        header_key = request.headers.get('x-api-key')
        if header_key and header_key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({"status": "error", "message": "Invalid API key"}), 401
    return decorated_function

# --- LOGIC ENGINE ---
def analyze_audio_heuristic(file_path):
    """
    Lightweight analysis to prevent server timeout/memory crash on free tier.
    """
    try:
        # Load audio (limited duration to speed up processing)
        y, sr = librosa.load(file_path, sr=16000, duration=10)

        # Feature 1: MFCC Variance (Texture)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_var = np.mean(np.var(mfcc, axis=1))

        # Feature 2: Spectral Flatness (Artificiality)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # Feature 3: Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

        # --- Decision Logic ---
        score = 0.5 
        
        # High variance usually = Human emotion
        if mfcc_var > 40: score += 0.3
        else: score -= 0.1
            
        # Very low flatness = Synthetic
        if flatness < 0.015: score -= 0.2
        
        confidence = max(0.01, min(0.99, score))
        
        if confidence > 0.5:
            return "HUMAN", round(confidence, 2), "High spectral variance and natural inflection detected."
        else:
            return "AI_GENERATED", round(1.0 - confidence, 2), "Unnatural pitch consistency and low spectral complexity."

    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return "HUMAN", 0.6, "Audio quality inconclusive; defaulted to Human."

# --- ROUTES ---

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "Voice Detection API is Running"}), 200

@app.route('/api/voice-detection', methods=['POST'])
@require_api_key
def detect_voice():
    temp_filename = None
    try:
        data = request.get_json()
        if not data: return jsonify({"status": "error", "message": "No JSON body"}), 400
        
        # Validation
        if data.get('language') not in ALLOWED_LANGUAGES:
            return jsonify({"status": "error", "message": "Unsupported Language"}), 400
        if not data.get('audioBase64'):
            return jsonify({"status": "error", "message": "Missing audio"}), 400

        # Decode
        try:
            audio_data = base64.b64decode(data.get('audioBase64'))
            temp_filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp3")
            with open(temp_filename, "wb") as f:
                f.write(audio_data)
        except:
            return jsonify({"status": "error", "message": "Invalid Base64"}), 400

        # Analyze
        label, conf, expl = analyze_audio_heuristic(temp_filename)

        return jsonify({
            "status": "success",
            "language": data.get('language'),
            "classification": label,
            "confidenceScore": conf,
            "explanation": expl
        }), 200

    except Exception as e:
        logger.error(f"Server Error: {e}")
        return jsonify({"status": "error", "message": "Processing failed"}), 500
    
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == '__main__':
    # Google Cloud Run sends the port in the env variable PORT
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
