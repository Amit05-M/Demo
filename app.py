"""
MoodBeats - Music Recommendation via Facial Expression OR Voice Emotion
Toggle between camera-based face detection or mic-based voice detection.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request

# ── Optional heavy deps ──────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except Exception:
    DEEPFACE_OK = False

try:
    from fer import FER
    FER_OK = True
except Exception:
    FER_OK = False

try:
    import librosa
    import numpy as np
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

# ── App setup ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
SONGS_PATH = ROOT / "data" / "songs.json"

app = Flask(__name__)

# ── Load songs ───────────────────────────────────────────────────────────────
with open(SONGS_PATH, encoding="utf-8") as f:
    ALL_SONGS: List[Dict] = json.load(f)

LOGGER.info("Loaded %d songs", len(ALL_SONGS))

# ── Emotion mapping ──────────────────────────────────────────────────────────
EMOTION_MAP = {
    "disgust": "angry",
    "fear": "surprise",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "surprise": "surprise",
    "surprised": "surprise",
}
VALID_EMOTIONS = {"happy", "sad", "angry", "neutral", "surprise"}


def normalize_emotion(label: str) -> str:
    label = label.strip().lower()
    return EMOTION_MAP.get(label, "neutral" if label not in VALID_EMOTIONS else label)


# ── Song recommendation ──────────────────────────────────────────────────────
def recommend_songs(emotion: str, language: str = "both", limit: int = 6) -> List[Dict]:
    emotion = normalize_emotion(emotion)
    pool = [s for s in ALL_SONGS if s["emotion"] == emotion]
    if language != "both":
        pool = [s for s in pool if s["language"].lower() == language.lower()]
    if not pool:
        pool = ALL_SONGS[:limit]
    random.shuffle(pool)
    return pool[:limit]


def extract_youtube_id(url: str) -> Optional[str]:
    if not url:
        return None
    url = url.strip()
    if "youtu.be/" in url:
        return url.split("youtu.be/", 1)[1].split("?")[0]
    if "youtube.com/watch" in url and "v=" in url:
        for part in url.split("?", 1)[-1].split("&"):
            if part.startswith("v="):
                return part[2:]
    if "youtube.com/embed/" in url:
        return url.split("embed/", 1)[1].split("?")[0]
    return None


def build_song_payload(song: Dict) -> Dict:
    yt_url = song.get("urls", {}).get("youtube", "")
    return {
        "title": song["title"],
        "artist": song["artist"],
        "emotion": song["emotion"],
        "language": song.get("language", ""),
        "tags": song.get("tags", []),
        "spotify": song.get("urls", {}).get("spotify", ""),
        "youtubeUrl": yt_url,
        "youtubeEmbedId": extract_youtube_id(yt_url),
    }


# ── Face emotion detection ───────────────────────────────────────────────────
def detect_face_emotion(image_data_url: str) -> Optional[Dict]:
    if not CV2_OK:
        LOGGER.warning("OpenCV not available – using mock face detection")
        return _mock_emotion()

    try:
        _, payload = image_data_url.split(",", 1)
        img_bytes = base64.b64decode(payload)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        # Try DeepFace first (most accurate)
        if DEEPFACE_OK:
            try:
                results = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=True,
                    detector_backend="opencv",
                    silent=True,
                )
                r = results[0] if isinstance(results, list) else results
                dominant = r["dominant_emotion"]
                score = r["emotion"][dominant] / 100.0
                return {"label": normalize_emotion(dominant), "score": round(score, 3), "source": "deepface"}
            except Exception as e:
                LOGGER.debug("DeepFace failed: %s, trying FER", e)

        # Fallback: FER
        if FER_OK:
            try:
                detector = FER(mtcnn=False)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = detector.detect_emotions(rgb)
                if detections:
                    top = max(detections, key=lambda d: max(d["emotions"].values()))
                    emotions = top["emotions"]
                    label, score = max(emotions.items(), key=lambda x: x[1])
                    return {"label": normalize_emotion(label), "score": round(float(score), 3), "source": "fer"}
            except Exception as e:
                LOGGER.debug("FER failed: %s", e)

        # Fallback: Haar cascade + brightness heuristic
        return _opencv_heuristic(frame)

    except Exception as e:
        LOGGER.error("Face detection error: %s", e)
        return None


def _opencv_heuristic(frame) -> Optional[Dict]:
    """Improved heuristic using Haar + multiple facial region features."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            LOGGER.debug("No face found in heuristic")
            return None

        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        # ── Features ─────────────────────────────────────────────────
        brightness = float(np.mean(face_roi))
        contrast   = float(np.std(face_roi))

        # Edge density (more edges = more expression / movement)
        edges = cv2.Canny(face_roi, 50, 150)
        edge_density = float(np.mean(edges > 0))

        # Laplacian variance (sharpness / texture richness)
        lap_var = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())

        # Upper vs lower half brightness ratio (raised brows = upper brighter)
        upper = float(np.mean(face_roi[:h//2, :]))
        lower = float(np.mean(face_roi[h//2:, :]))
        ul_ratio = upper / (lower + 1e-6)

        LOGGER.info("Face heuristic: brightness=%.1f contrast=%.1f edges=%.3f lap=%.1f ul=%.2f",
                    brightness, contrast, edge_density, lap_var, ul_ratio)

        # ── Scoring ──────────────────────────────────────────────────
        scores = {"happy": 0.0, "sad": 0.0, "angry": 0.0, "surprise": 0.0, "neutral": 0.0}

        # Brightness: happy faces tend brighter, sad/angry darker
        if brightness > 140:
            scores["happy"]    += 2.0
        elif brightness > 115:
            scores["happy"]    += 1.0
            scores["neutral"]  += 1.0
        elif brightness > 90:
            scores["neutral"]  += 1.5
            scores["sad"]      += 0.5
        else:
            scores["sad"]      += 1.5
            scores["angry"]    += 1.0

        # Contrast: angry/surprised faces have more muscle tension → more contrast
        if contrast > 60:
            scores["angry"]    += 1.5
            scores["surprise"] += 1.0
        elif contrast > 45:
            scores["happy"]    += 1.0
            scores["neutral"]  += 1.0
        else:
            scores["sad"]      += 1.5
            scores["neutral"]  += 0.5

        # Edge density: expressive faces have more edges
        if edge_density > 0.15:
            scores["surprise"] += 2.0
            scores["angry"]    += 1.0
        elif edge_density > 0.08:
            scores["happy"]    += 1.0
        else:
            scores["sad"]      += 1.0
            scores["neutral"]  += 1.0

        # Laplacian variance: surprise/angry = sharp textures
        if lap_var > 600:
            scores["surprise"] += 1.5
            scores["angry"]    += 1.0
        elif lap_var > 300:
            scores["happy"]    += 1.0
        else:
            scores["sad"]      += 0.5
            scores["neutral"]  += 1.0

        # Upper/lower ratio: raised brows (surprise) = upper brighter
        if ul_ratio > 1.08:
            scores["surprise"] += 1.5
        elif ul_ratio < 0.95:
            scores["angry"]    += 1.0

        label = max(scores, key=lambda k: scores[k])
        top = scores[label]
        total = sum(scores.values()) or 1.0
        score = float(np.clip(0.45 + (top / total) * 0.45, 0.45, 0.85))

        LOGGER.info("Face heuristic result: %s (%.0f%%)", label, score * 100)
        return {"label": label, "score": score, "source": "opencv_heuristic"}

    except Exception as e:
        LOGGER.error("OpenCV heuristic error: %s", e)
        return None


def _mock_emotion() -> Dict:
    emotions = ["happy", "sad", "neutral", "angry", "surprise"]
    label = random.choice(emotions)
    return {"label": label, "score": round(random.uniform(0.55, 0.90), 2), "source": "mock"}


# ── Voice emotion detection ──────────────────────────────────────────────────
def detect_voice_emotion(audio_data_url: str) -> Optional[Dict]:
    if not LIBROSA_OK:
        LOGGER.warning("librosa not available – using mock voice detection")
        return _mock_emotion()

    try:
        _, payload = audio_data_url.split(",", 1)
        audio_bytes = base64.b64decode(payload)
        with io.BytesIO(audio_bytes) as buf:
            y, sr = librosa.load(buf, sr=22050, mono=True)

        if y.size == 0:
            return None

        # ── Trim silence, normalize ─────────────────────────────────────
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if y_trimmed.size < sr * 0.3:          # less than 0.3s of voice → very quiet
            y_trimmed = y

        # Normalize amplitude so RMS thresholds are scale-independent
        peak = float(np.max(np.abs(y_trimmed)))
        if peak > 1e-6:
            y_norm = y_trimmed / peak
        else:
            return {"label": "neutral", "score": 0.45, "source": "silence"}

        # ── Feature extraction ──────────────────────────────────────────
        tempo_arr, _ = librosa.beat.beat_track(y=y_norm, sr=sr)
        tempo = float(tempo_arr)

        rms_raw = float(np.mean(librosa.feature.rms(y=y_norm)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_norm, sr=sr)))
        zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y_norm)))
        rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y_norm, sr=sr)))

        # MFCCs help distinguish speech energy patterns
        mfccs = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=13)
        mfcc_mean = float(np.mean(mfccs[1]))   # 2nd MFCC correlates with vocal energy

        LOGGER.info("Voice features: tempo=%.1f rms=%.4f centroid=%.0f zcr=%.4f mfcc1=%.2f",
                    tempo, rms_raw, centroid, zcr, mfcc_mean)

        # ── Scoring each emotion (higher = more likely) ─────────────────
        # All features are now on normalized audio, so rms is typically 0.1–0.6
        scores: Dict[str, float] = {
            "happy":    0.0,
            "sad":      0.0,
            "angry":    0.0,
            "surprise": 0.0,
            "neutral":  0.0,
        }

        # Tempo cues
        if tempo > 120:
            scores["happy"]    += 2.0
            scores["angry"]    += 1.0
        elif tempo > 95:
            scores["happy"]    += 1.0
            scores["neutral"]  += 1.5
        else:
            scores["sad"]      += 2.0
            scores["neutral"]  += 1.0

        # Spectral centroid (brightness of sound)
        if centroid > 3000:
            scores["angry"]    += 2.0
            scores["surprise"] += 1.5
        elif centroid > 2000:
            scores["happy"]    += 1.5
            scores["neutral"]  += 1.0
        else:
            scores["sad"]      += 2.0

        # Zero crossing rate (roughness / harshness)
        if zcr > 0.12:
            scores["angry"]    += 2.0
            scores["surprise"] += 1.0
        elif zcr > 0.07:
            scores["happy"]    += 1.0
            scores["neutral"]  += 1.0
        else:
            scores["sad"]      += 1.5
            scores["neutral"]  += 0.5

        # MFCC energy distribution
        if mfcc_mean > 5:
            scores["happy"]    += 1.5
            scores["angry"]    += 1.0
        elif mfcc_mean < -5:
            scores["sad"]      += 1.5
        else:
            scores["neutral"]  += 1.5

        # Rolloff
        if rolloff > 4000:
            scores["angry"]    += 1.0
            scores["surprise"] += 1.0
        elif rolloff < 2000:
            scores["sad"]      += 1.0

        label = max(scores, key=lambda k: scores[k])
        top_score = scores[label]
        total = sum(scores.values()) or 1.0
        confidence = float(np.clip(0.45 + (top_score / total) * 0.5, 0.45, 0.92))

        return {
            "label": normalize_emotion(label),
            "score": round(confidence, 3),
            "source": "librosa",
            "features": {
                "tempo": round(tempo, 1),
                "rms": round(rms_raw, 4),
                "centroid": round(centroid, 1),
                "zcr": round(zcr, 4),
                "rolloff": round(rolloff, 1),
                "scores": {k: round(v, 2) for k, v in scores.items()},
            },
        }
    except Exception as e:
        LOGGER.error("Voice detection error: %s", e)
        return None


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "camera")       # "camera" or "voice"
    photo = data.get("photo")
    audio = data.get("audio")
    language = data.get("language", "both")

    emotion_result = None

    if mode == "camera" and photo:
        emotion_result = detect_face_emotion(photo)
    elif mode == "voice" and audio:
        emotion_result = detect_voice_emotion(audio)

    if emotion_result is None:
        emotion_result = {"label": "neutral", "score": 0.5, "source": "fallback"}

    label = emotion_result["label"]
    songs = recommend_songs(label, language=language, limit=6)

    return jsonify({
        "emotion": label,
        "confidence": emotion_result.get("score", 0.5),
        "source": emotion_result.get("source", "unknown"),
        "features": emotion_result.get("features"),
        "songs": [build_song_payload(s) for s in songs],
    })


@app.route("/songs")
def songs_list():
    """Debug endpoint — list all songs."""
    return jsonify(ALL_SONGS)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
