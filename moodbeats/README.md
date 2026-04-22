# 🎵 MoodBeats — Emotion-Powered Music Recommendation

Detect your mood using **Camera** (facial expression) **OR** **Voice** (microphone audio), and get instant Hindi & Marathi song recommendations.

---

## ✨ Features

- **Two detection modes** — switch between Camera and Voice with one click
- **Real emotion detection** — Uses DeepFace (face) + Librosa (voice)
- **50+ songs** — Hindi & Marathi across 5 emotions: happy, sad, angry, neutral, surprise
- **YouTube embeds** — Watch/listen right in the page
- **Spotify + YouTube links** — Open in your favourite player
- **Language filter** — Show Hindi, Marathi, or both

---

## 🔧 Setup & Run

### 1. Install Python dependencies

```bash
pip install flask opencv-python-headless numpy deepface fer librosa soundfile
```

> **Note on DeepFace**: First run will automatically download the emotion model (~100 MB). This only happens once.

> **Minimal install** (if DeepFace fails): The app has graceful fallbacks. Even with only `flask` and `opencv-python-headless`, it will work using an OpenCV heuristic for face detection and mock results for voice.

### 2. Run the app

```bash
cd moodbeats
python app.py
```

### 3. Open in browser

```
http://localhost:5000
```

---

## 🎭 How It Works

### Camera Mode
1. Clicks "Detect My Mood"
2. Browser opens your webcam
3. App captures a frame after 2 seconds
4. Frame is sent to Flask backend
5. **DeepFace** (or FER / OpenCV fallback) detects emotion
6. Songs matching the emotion are shown with YouTube embeds

### Voice Mode
1. Clicks "Record & Detect"
2. Browser opens your microphone
3. App records 5 seconds of audio
4. Audio is sent to Flask backend
5. **Librosa** extracts features (tempo, RMS energy, spectral centroid, ZCR)
6. Rule-based classifier maps features → emotion
7. Songs matching the emotion are shown

---

## 📁 Project Structure

```
moodbeats/
├── app.py                  ← Flask backend (emotion detection + song recommendation)
├── requirements.txt
├── data/
│   └── songs.json          ← 50+ Hindi & Marathi songs with YouTube/Spotify URLs
├── templates/
│   └── index.html          ← Main UI
└── static/
    ├── style.css            ← Cinematic dark theme
    └── script.js            ← Camera/voice capture + API calls
```

---

## 🎵 Songs Included

**Emotions covered:** Happy, Sad, Angry, Surprise, Neutral

**Hindi songs:** Zinda, Tum Hi Ho, Channa Mereya, Badtameez Dil, Nagada Sang Dhol, Deva Shree Ganesha, Ghoomar, Lag Ja Gale, Kun Faya Kun, Raabta, and more…

**Marathi songs:** Zingaat, Apsara Aali, Natrang, Vaajle Ki Baara, Powada Shivaji Maharaj, Tujhya Rupacha, Man Tarpat, and more…

---

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| Camera/mic permission denied | Allow in browser settings (lock icon in URL bar) |
| DeepFace download fails | Run `pip install deepface` separately; the app will use OpenCV fallback |
| `librosa` not found | `pip install librosa soundfile` |
| Voice always says "neutral" | Speak clearly; quieter environments work better |
| Black video feed | Try a different browser (Chrome recommended) |

---

## 🔑 Environment Variables (Optional)

| Variable | Purpose |
|---|---|
| `PORT` | Change port (default: 5000) |
| `FLASK_ENV` | Set to `production` to disable debug mode |
