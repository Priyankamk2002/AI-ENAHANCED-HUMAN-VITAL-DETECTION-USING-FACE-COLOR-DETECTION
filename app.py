import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vital Sign Monitor",
    page_icon="💓",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0a0a0f; color: #e8e8f0; }
  .stApp { background-color: #0a0a0f; }
  .header-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.4rem;
    background: linear-gradient(135deg, #ff4d6d, #ff8fa3, #c77dff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;
  }
  .header-sub {
    font-family: 'Space Mono', monospace; font-size: 0.78rem; color: #666680;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 2rem;
  }
  .metric-card {
    background: linear-gradient(145deg, #13131f, #1a1a2e); border: 1px solid #2a2a3e;
    border-radius: 16px; padding: 1.5rem 2rem; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 1rem;
  }
  .metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #ff4d6d, #c77dff);
  }
  .metric-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: #666680; margin-bottom: 0.5rem; }
  .metric-value { font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700; line-height: 1; }
  .metric-unit { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #888899; margin-top: 0.3rem; }
  .metric-hr .metric-value { color: #ff4d6d; }
  .metric-spo2 .metric-value { color: #7ec8e3; }
  .status-badge {
    display: inline-block; font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.12em; text-transform: uppercase; padding: 4px 12px;
    border-radius: 20px; margin-top: 0.6rem;
  }
  .badge-normal { background: #0d2b1d; color: #4ade80; border: 1px solid #16a34a; }
  .badge-warning { background: #2b1e0d; color: #fbbf24; border: 1px solid #d97706; }
  .badge-measuring { background: #1a1a2e; color: #818cf8; border: 1px solid #4f46e5; }
  .info-box {
    background: #13131f; border: 1px solid #2a2a3e; border-radius: 12px;
    padding: 1rem 1.2rem; font-family: 'Space Mono', monospace;
    font-size: 0.72rem; color: #888899; line-height: 1.7; margin-top: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ─── Signal Processing ───────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def bandpass_filter(data, lowcut=0.8, highcut=2.5, fs=30, order=5):
    if len(data) < order * 3:
        return np.zeros_like(data)
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_heart_rate(signal, fs=30):
    if len(signal) < fs * 5:
        return None
    signal_arr = np.array(signal)
    signal_norm = (signal_arr - np.mean(signal_arr)) / (np.std(signal_arr) + 1e-8)
    filtered = bandpass_filter(signal_norm, 0.8, 2.5, fs)
    peaks, _ = find_peaks(filtered, distance=fs // 2, height=np.percentile(filtered, 80))
    if len(peaks) < 2:
        return None
    avg_interval = np.median(np.diff(peaks) / fs)
    bpm = int(60 / avg_interval)
    return bpm if 50 <= bpm <= 120 else None

def calculate_spo2(red_signal, green_signal):
    if len(red_signal) < 10 or len(green_signal) < 10:
        return None
    ac_red, dc_red = np.std(red_signal), np.mean(red_signal)
    ac_green, dc_green = np.std(green_signal), np.mean(green_signal)
    if dc_red == 0 or dc_green == 0:
        return None
    r_ratio = (ac_red / dc_red) / (ac_green / dc_green)
    spo2 = 104 - (20 * r_ratio)
    return max(90, min(100, int(spo2)))

# ─── Video Processor ────────────────────────────────────────────────────────
class VitalProcessor(VideoProcessorBase):
    def __init__(self):
        self.green_buffer = deque(maxlen=300)
        self.red_buffer = deque(maxlen=300)
        self.heart_rate = None
        self.spo2 = None
        self.fs = 30

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 80, 110), 2)
            cv2.putText(img, "FACE DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 110), 1)

            cheek = img[y + h // 3: y + 2 * h // 3, x + w // 4: x + 3 * w // 4]
            if cheek.size > 0:
                self.green_buffer.append(np.mean(cheek[:, :, 1]))
                self.red_buffer.append(np.mean(cheek[:, :, 2]))

                if len(self.green_buffer) >= self.fs * 5:
                    hr = calculate_heart_rate(list(self.green_buffer), self.fs)
                    spo2 = calculate_spo2(list(self.red_buffer), list(self.green_buffer))
                    if hr: self.heart_rate = hr
                    if spo2: self.spo2 = spo2

            hr_text = f"HR: {self.heart_rate} BPM" if self.heart_rate else "HR: Measuring..."
            spo2_text = f"SpO2: {self.spo2}%" if self.spo2 else "SpO2: Measuring..."
            cv2.putText(img, hr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 110), 2)
            cv2.putText(img, spo2_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (126, 200, 227), 2)
        else:
            cv2.putText(img, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 200), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="header-title">💓 Vital Sign Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Non-contact · Real-time · Camera-based rPPG</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    ctx = webrtc_streamer(
        key="vital-monitor",
        video_processor_factory=VitalProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
    st.markdown(
        '<div class="info-box">'
        '⚡ Click START above to allow camera access<br>'
        '💡 Good lighting improves accuracy<br>'
        '⏱ Wait 5–10 sec for readings to stabilise<br>'
        '📏 Sit 40–60 cm from camera'
        '</div>', unsafe_allow_html=True
    )

with right_col:
    hr_placeholder = st.empty()
    spo2_placeholder = st.empty()

    def render_metric(placeholder, label, value, unit, css_class, status_class, status_text):
        placeholder.markdown(
            f'<div class="metric-card {css_class}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value if value else "—"}</div>'
            f'<div class="metric-unit">{unit}</div>'
            f'<span class="status-badge badge-{status_class}">{status_text}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    if ctx.video_processor:
        hr = ctx.video_processor.heart_rate
        spo2 = ctx.video_processor.spo2
        hr_s = "normal" if hr and 60 <= hr <= 100 else ("warning" if hr else "measuring")
        hr_t = "NORMAL" if hr and 60 <= hr <= 100 else ("HIGH/LOW" if hr else "MEASURING")
        sp_s = "normal" if spo2 and spo2 >= 95 else ("warning" if spo2 else "measuring")
        sp_t = "NORMAL" if spo2 and spo2 >= 95 else ("LOW" if spo2 else "MEASURING")
        render_metric(hr_placeholder, "HEART RATE", hr, "beats per minute", "metric-hr", hr_s, hr_t)
        render_metric(spo2_placeholder, "SpO₂", spo2, "oxygen saturation", "metric-spo2", sp_s, sp_t)
    else:
        render_metric(hr_placeholder, "HEART RATE", None, "beats per minute", "metric-hr", "measuring", "WAITING")
        render_metric(spo2_placeholder, "SpO₂", None, "oxygen saturation", "metric-spo2", "measuring", "WAITING")



       
