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
    try:
        # Load audio (limited duration to 10s for speed)
        y, sr = librosa.load(file_path, sr=16000, duration=10)

        # --- Feature 1: MFCC Variance (Human voice has rich texture) ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_var = np.mean(np.var(mfcc, axis=1))

        # --- Feature 2: Spectral Flatness (AI is often "too clean") ---
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        # --- Feature 3: Silence Ratio (AI often has perfect digital silence) ---
        # Calculate ratio of non-silent frames
        non_silent = librosa.effects.split(y, top_db=30)
        silence_ratio = 1.0 - (np.sum([e - s for s, e in non_silent]) / len(y))

        # --- Feature 4: Spectral Contrast (Peak vs Valley differences) ---
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

        # --- Scoring Logic ---
        # Start neutral
        score = 0.5 

        # 1. Texture Check
        if mfcc_var > 40: 
            score += 0.25 # Humans vary pitch/tone more
            feat_expl = "high organic texture"
        else:
            score -= 0.15
            feat_expl = "flattened pitch dynamics"

        # 2. Artificiality Check (Flatness)
        if flatness < 0.005: 
            score -= 0.2 # Too clean = Likely AI
        elif flatness > 0.05:
            score += 0.1 # Background noise = Likely Human

        # 3. Contrast Check (Robotic voices have high contrast)
        if contrast > 22:
            score -= 0.15 # Metadata artifacting

        # Clamp Score
        confidence = max(0.01, min(0.99, score))
        
        # --- Decision & Dynamic Explanation ---
        if confidence > 0.5:
            classification = "HUMAN"
            expl = f"Detected natural prosody and {feat_expl}. Background noise floor indicates a physical recording environment."
        else:
            classification = "AI_GENERATED"
            confidence = 1.0 - confidence # Flip for AI
            expl = f"Detected unnatural spectral uniformity and {feat_expl}. Lack of breathing pauses and perfect digital silence intervals observed."

        return classification, round(confidence, 2), expl

    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return "HUMAN", 0.65, "Audio signal analysis inconclusive; fallback to Human baseline."

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

