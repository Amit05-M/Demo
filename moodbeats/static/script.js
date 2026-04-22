/* ── MoodBeats Frontend Script ──────────────────────────────────────────── */

// ── DOM refs ─────────────────────────────────────────────────────────────
const detectBtn        = document.getElementById('detectBtn');
const detectLabel      = document.getElementById('detectLabel');
const btnCamera        = document.getElementById('btnCamera');
const btnVoice         = document.getElementById('btnVoice');
const cameraArea       = document.getElementById('cameraArea');
const voiceArea        = document.getElementById('voiceArea');
const videoEl          = document.getElementById('videoEl');
const camPlaceholder   = document.getElementById('camPlaceholder');
const scanOverlay      = document.getElementById('scanOverlay');
const camStatus        = document.getElementById('camStatus');
const voiceStatus      = document.getElementById('voiceStatus');
const micRing          = document.getElementById('micRing');
const waveBars         = document.getElementById('waveBars');
const voicePlaceholder = document.getElementById('voicePlaceholder');
const voiceRecBadge    = document.getElementById('voiceRecordingBadge');
const loadingLayer     = document.getElementById('loadingLayer');
const resultsSection   = document.getElementById('resultsSection');
const emotionEmoji     = document.getElementById('emotionEmoji');
const emotionLabel     = document.getElementById('emotionLabel');
const emotionMeta      = document.getElementById('emotionMeta');
const songGrid         = document.getElementById('songGrid');
const errorCard        = document.getElementById('errorCard');
const errorMsg         = document.getElementById('errorMsg');
const langBtns         = document.querySelectorAll('.lang-btn');

// ── State ─────────────────────────────────────────────────────────────────
let currentMode     = 'camera';  // 'camera' | 'voice'
let selectedLang    = 'both';
let activeStream    = null;
let isDetecting     = false;

// Emotion → emoji + color mapping
const EMOTION_META = {
  happy:    { emoji: '😄', color: '#fbbf24', label: 'Happy' },
  sad:      { emoji: '😢', color: '#60a5fa', label: 'Sad' },
  angry:    { emoji: '😠', color: '#f87171', label: 'Angry' },
  surprise: { emoji: '😲', color: '#a78bfa', label: 'Surprised' },
  neutral:  { emoji: '😐', color: '#9ca3af', label: 'Neutral' },
  fear:     { emoji: '😨', color: '#94a3b8', label: 'Fearful' },
  disgust:  { emoji: '🤢', color: '#86efac', label: 'Disgusted' },
};

// ── Mode Toggle ───────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;

  btnCamera.classList.toggle('active', mode === 'camera');
  btnVoice.classList.toggle('active', mode === 'voice');
  cameraArea.classList.toggle('hidden', mode !== 'camera');
  voiceArea.classList.toggle('hidden', mode !== 'voice');

  detectLabel.textContent = mode === 'camera' ? 'Detect My Mood' : 'Record & Detect';

  // Reset states
  hideError();
  resultsSection.classList.add('hidden');
}

btnCamera.addEventListener('click', () => setMode('camera'));
btnVoice.addEventListener('click', () => setMode('voice'));

// ── Language Toggle ────────────────────────────────────────────────────────
langBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    langBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedLang = btn.dataset.lang;
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────
function setStatus(msg, isCamera = true) {
  (isCamera ? camStatus : voiceStatus).textContent = msg;
}

function showLoading(on) {
  loadingLayer.classList.toggle('hidden', !on);
  detectBtn.disabled = on;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorCard.classList.remove('hidden');
}

function hideError() {
  errorCard.classList.add('hidden');
}

function stopStream() {
  if (activeStream) {
    activeStream.getTracks().forEach(t => t.stop());
    activeStream = null;
  }
  videoEl.srcObject = null;
}

function takeSnapshot() {
  const c = document.createElement('canvas');
  c.width = videoEl.videoWidth || 640;
  c.height = videoEl.videoHeight || 480;
  c.getContext('2d').drawImage(videoEl, 0, 0);
  return c.toDataURL('image/png');
}

// ── WAV encoder ────────────────────────────────────────────────────────────
function encodeWav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true);
  ws(8, 'WAVE'); ws(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); ws(36, 'data');
  v.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function flattenBuffers(bufs) {
  const total = bufs.reduce((a, b) => a + b.length, 0);
  const out = new Float32Array(total);
  let off = 0;
  bufs.forEach(b => { out.set(b, off); off += b.length; });
  return out;
}

async function recordAudio(stream, seconds = 6) {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return null;
  const ctx = new AC();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const bufs = [];
  proc.onaudioprocess = e => bufs.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(ctx.destination);
  await new Promise(r => setTimeout(r, seconds * 1000));
  src.disconnect(); proc.disconnect(); await ctx.close();
  const wav = encodeWav(flattenBuffers(bufs), ctx.sampleRate || 22050);
  return wav;
}

function blobToDataUrl(blob) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onloadend = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(blob);
  });
}

// ── API call ───────────────────────────────────────────────────────────────
async function callAnalyze(payload) {
  const res = await fetch('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}

// ── Song Card Builder ─────────────────────────────────────────────────────
function buildSongCard(song) {
  const embed = song.youtubeEmbedId
    ? `<div class="song-embed">
         <iframe src="https://www.youtube.com/embed/${song.youtubeEmbedId}?rel=0&modestbranding=1"
                 allowfullscreen loading="lazy" title="${song.title}"></iframe>
       </div>`
    : `<div class="no-embed">No YouTube preview</div>`;

  const spotifyBtn = song.spotify
    ? `<a href="${song.spotify}" target="_blank" rel="noreferrer" class="btn-link btn-spotify">▷ Spotify</a>`
    : '';
  const ytBtn = song.youtubeUrl
    ? `<a href="${song.youtubeUrl}" target="_blank" rel="noreferrer" class="btn-link btn-youtube">▷ YouTube</a>`
    : '';

  const tags = (song.tags || []).map(t => `<span class="song-tag">${t}</span>`).join('');
  const langBadge = song.language ? `<span class="song-lang">${song.language}</span>` : '';

  return `
    <div class="song-card">
      ${embed}
      <div class="song-body">
        <div>
          <div class="song-title">${song.title}</div>
          <div class="song-artist">${song.artist}</div>
        </div>
        <div class="song-tags">${tags}${langBadge}</div>
      </div>
      <div class="song-links">${spotifyBtn}${ytBtn}</div>
    </div>`;
}

// ── Render results ────────────────────────────────────────────────────────
function renderResults(data) {
  const meta = EMOTION_META[data.emotion] || EMOTION_META.neutral;

  emotionEmoji.textContent = meta.emoji;
  emotionLabel.textContent = meta.label;
  emotionLabel.style.backgroundImage = `linear-gradient(135deg, ${meta.color}, #c084fc)`;

  const confPct = Math.round((data.confidence || 0) * 100);
  const srcMap = { deepface: 'DeepFace AI', fer: 'FER Model', opencv_heuristic: 'OpenCV', librosa: 'Audio Analysis', mock: 'Simulated', fallback: 'Default', silence: 'Silence detected' };
  const srcLabel = srcMap[data.source] || data.source || '—';

  let featureStr = '';
  if (data.features) {
    const f = data.features;
    if (f.tempo) featureStr += ` · tempo ${f.tempo}`;
    if (f.centroid) featureStr += ` · centroid ${Math.round(f.centroid)}`;
    if (f.scores) {
      const sorted = Object.entries(f.scores).sort((a,b) => b[1]-a[1]).slice(0,3);
      featureStr += ` · top: ${sorted.map(([k,v]) => `${k}(${v})`).join(', ')}`;
    }
  }

  emotionMeta.textContent = `${confPct}% confidence · ${srcLabel}${featureStr}`;

  if (!data.songs || data.songs.length === 0) {
    songGrid.innerHTML = '<p style="color:var(--muted);grid-column:1/-1">No songs found for this mood. Try again!</p>';
  } else {
    songGrid.innerHTML = data.songs.map(buildSongCard).join('');
  }

  resultsSection.classList.remove('hidden');
}

// ── Camera Flow ───────────────────────────────────────────────────────────
async function runCameraFlow() {
  setStatus('Requesting camera access…', true);

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    audio: false,
  });
  activeStream = stream;
  videoEl.srcObject = stream;
  camPlaceholder.classList.add('hidden');

  await new Promise(res => {
    const t = setTimeout(res, 3000);
    videoEl.onloadeddata = () => { clearTimeout(t); res(); };
  });
  await videoEl.play().catch(() => {});

  setStatus('Camera active — capturing your expression…', true);
  scanOverlay.classList.remove('hidden');
  await new Promise(r => setTimeout(r, 2000)); // let face settle

  const photo = takeSnapshot();
  scanOverlay.classList.add('hidden');

  stopStream();
  camPlaceholder.classList.remove('hidden');
  setStatus('Photo captured — analyzing…', true);

  return { mode: 'camera', photo, language: selectedLang };
}

// ── Voice Flow ────────────────────────────────────────────────────────────
async function runVoiceFlow() {
  setStatus('Requesting microphone access…', false);

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  activeStream = stream;
  voicePlaceholder.classList.add('hidden');
  micRing.classList.add('recording');
  waveBars.classList.add('active');
  voiceRecBadge.classList.remove('hidden');

  const DURATION = 6;

  // Start recording AND countdown simultaneously
  const recordPromise = recordAudio(stream, DURATION);

  for (let i = DURATION; i >= 1; i--) {
    setStatus(`🎙 Recording… ${i}s — speak, sing, or hum something!`, false);
    await new Promise(r => setTimeout(r, 1000));
  }

  const audioBlob = await recordPromise;

  micRing.classList.remove('recording');
  waveBars.classList.remove('active');
  voiceRecBadge.classList.add('hidden');
  stopStream();
  voicePlaceholder.classList.remove('hidden');
  setStatus('Audio captured — analyzing…', false);

  const audioData = audioBlob ? await blobToDataUrl(audioBlob) : null;
  return { mode: 'voice', audio: audioData, language: selectedLang };
}

// ── Main Detection ────────────────────────────────────────────────────────
async function runDetection() {
  if (isDetecting) return;
  isDetecting = true;
  hideError();
  resultsSection.classList.add('hidden');
  showLoading(true);

  try {
    let payload;
    if (currentMode === 'camera') {
      payload = await runCameraFlow();
    } else {
      payload = await runVoiceFlow();
    }

    const result = await callAnalyze(payload);
    renderResults(result);

  } catch (err) {
    console.error('Detection error:', err);
    let msg = err.message || 'Unknown error';
    if (msg.includes('Permission') || msg.includes('NotAllowed') || msg.includes('permission')) {
      msg = `${currentMode === 'camera' ? 'Camera' : 'Microphone'} access denied. Please allow access in browser settings and try again.`;
    } else if (msg.includes('NotFound') || msg.includes('DevicesNotFound')) {
      msg = `No ${currentMode === 'camera' ? 'camera' : 'microphone'} found. Please connect a device and try again.`;
    }
    showError(msg);
    stopStream();
    camPlaceholder.classList.remove('hidden');
    voicePlaceholder.classList.remove('hidden');
    micRing.classList.remove('recording');
    waveBars.classList.remove('active');
    voiceRecBadge.classList.add('hidden');
    scanOverlay.classList.add('hidden');

  } finally {
    showLoading(false);
    isDetecting = false;
    setStatus('Ready — click Detect My Mood', true);
    setStatus('Ready — click Detect My Mood', false);
  }
}

detectBtn.addEventListener('click', runDetection);
