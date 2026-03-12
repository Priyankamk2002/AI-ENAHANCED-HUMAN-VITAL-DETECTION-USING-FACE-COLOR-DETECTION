import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
import time

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

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
  }

  .stApp { background-color: #0a0a0f; }

  h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

  .header-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #ff4d6d, #ff8fa3, #c77dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }

  .header-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #666680;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  .metric-card {
    background: linear-gradient(145deg, #13131f, #1a1a2e);
    border: 1px solid #2a2a3e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }

  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #ff4d6d, #c77dff);
  }

  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #666680;
    margin-bottom: 0.5rem;
  }

  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
  }

  .metric-unit {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #888899;
    margin-top: 0.3rem;
  }

  .metric-hr .metric-value { color: #ff4d6d; }
  .metric-spo2 .metric-value { color: #7ec8e3; }

  .status-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-top: 0.6rem;
  }

  .badge-normal { background: #0d2b1d; color: #4ade80; border: 1px solid #16a34a; }
  .badge-warning { background: #2b1e0d; color: #fbbf24; border: 1px solid #d97706; }
  .badge-measuring { background: #1a1a2e; color: #818cf8; border: 1px solid #4f46e5; }

  .info-box {
    background: #13131f;
    border: 1px solid #2a2a3e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #888899;
    line-height: 1.7;
  }

  .stButton > button {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.08em;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.8rem;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100%;
  }

  div[data-testid="column"]:nth-child(1) .stButton > button {
    background: linear-gradient(135deg, #ff4d6d, #c77dff);
    color: white;
  }

  div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #1a1a2e;
    color: #888899;
    border: 1px solid #2a2a3e;
  }

  .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

  .divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
  }

  /* Chart styling */
  .js-plotly-plot .plotly { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Signal Processing Functions ────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def bandpass_filter(data, lowcut=0.8, highcut=2.5, fs=30, order=5):
    if len(data) < order * 3:
        return np.zeros_like(data)
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_roi(frame, face):
    x, y, w, h = face
    cheek_region = frame[y + h // 3: y + 2 * h // 3, x + w // 4: x + 3 * w // 4]
    return cheek_region if cheek_region.size > 0 else None

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

def hr_status(bpm):
    if bpm is None:
        return "measuring", "MEASURING"
    if bpm < 60:
        return "warning", "LOW"
    if bpm > 100:
        return "warning", "HIGH"
    return "normal", "NORMAL"

def spo2_status(spo2):
    if spo2 is None:
        return "measuring", "MEASURING"
    if spo2 < 95:
        return "warning", "LOW"
    return "normal", "NORMAL"

# ─── Session State ───────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False
if "green_buffer" not in st.session_state:
    st.session_state.green_buffer = deque(maxlen=300)
if "red_buffer" not in st.session_state:
    st.session_state.red_buffer = deque(maxlen=300)
if "heart_rate" not in st.session_state:
    st.session_state.heart_rate = None
if "spo2" not in st.session_state:
    st.session_state.spo2 = None

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="header-title">💓 Vital Sign Monitor</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Non-contact · Real-time · Camera-based rPPG</div>', unsafe_allow_html=True)

# ─── Layout ─────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # Camera feed
    cam_placeholder = st.empty()
    cam_placeholder.markdown(
        '<div style="background:#13131f;border:1px solid #2a2a3e;border-radius:16px;'
        'height:380px;display:flex;align-items:center;justify-content:center;'
        'color:#444455;font-family:Space Mono,monospace;font-size:0.8rem;letter-spacing:0.1em;">'
        '[ CAMERA FEED ]</div>', unsafe_allow_html=True
    )

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        start_btn = st.button("▶  Start", key="start")
    with btn_c2:
        stop_btn = st.button("■  Stop", key="stop")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        '⚡ Keep face centred in frame<br>'
        '💡 Good lighting improves accuracy<br>'
        '⏱ Allow 5–10 sec for readings to stabilise<br>'
        '📏 Sit ~40–60 cm from camera'
        '</div>', unsafe_allow_html=True
    )

with right_col:
    hr_placeholder = st.empty()
    spo2_placeholder = st.empty()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    chart_placeholder = st.empty()
    chart_placeholder.markdown(
        '<div style="background:#13131f;border:1px solid #2a2a3e;border-radius:12px;'
        'height:180px;display:flex;align-items:center;justify-content:center;'
        'color:#444455;font-family:Space Mono,monospace;font-size:0.75rem;letter-spacing:0.08em;">'
        '[ SIGNAL CHART ]</div>', unsafe_allow_html=True
    )

def render_metric(placeholder, label, value, unit, css_class, status_class, status_text):
    placeholder.markdown(
        f'<div class="metric-card {css_class}">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value">{value if value else "—"}</div>'
        f'  <div class="metric-unit">{unit}</div>'
        f'  <span class="status-badge badge-{status_class}">{status_text}</span>'
        f'</div><br>',
        unsafe_allow_html=True
    )

# Initial render
hs, ht = hr_status(None)
ss, st_ = spo2_status(None)
render_metric(hr_placeholder, "HEART RATE", None, "beats per minute", "metric-hr", hs, ht)
render_metric(spo2_placeholder, "SpO₂", None, "oxygen saturation", "metric-spo2", ss, st_)

# ─── Button Logic ────────────────────────────────────────────────────────────
if start_btn:
    st.session_state.running = True
    st.session_state.green_buffer.clear()
    st.session_state.red_buffer.clear()
    st.session_state.heart_rate = None
    st.session_state.spo2 = None

if stop_btn:
    st.session_state.running = False

# ─── Main Loop ───────────────────────────────────────────────────────────────
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check permissions and try again.")
        st.session_state.running = False
    else:
        fs = 30
        frame_count = 0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 80, 110), 2)
                cv2.putText(frame, "FACE DETECTED", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 110), 1)

                roi = extract_roi(frame, faces[0])
                if roi is not None and roi.shape[0] > 0:
                    st.session_state.green_buffer.append(np.mean(roi[:, :, 1]))
                    st.session_state.red_buffer.append(np.mean(roi[:, :, 2]))

                    if len(st.session_state.green_buffer) >= fs * 5:
                        hr = calculate_heart_rate(list(st.session_state.green_buffer), fs)
                        spo2 = calculate_spo2(
                            list(st.session_state.red_buffer),
                            list(st.session_state.green_buffer)
                        )
                        if hr:
                            st.session_state.heart_rate = hr
                        if spo2:
                            st.session_state.spo2 = spo2
            else:
                cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 200), 2)

            # Display frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(rgb, channels="RGB", use_container_width=True)

            # Update metrics every 15 frames
            if frame_count % 15 == 0:
                hr_val = st.session_state.heart_rate
                spo2_val = st.session_state.spo2
                hs, ht = hr_status(hr_val)
                ss, st_ = spo2_status(spo2_val)
                render_metric(hr_placeholder, "HEART RATE", hr_val, "beats per minute", "metric-hr", hs, ht)
                render_metric(spo2_placeholder, "SpO₂", spo2_val, "oxygen saturation", "metric-spo2", ss, st_)

                # Update chart
                if len(st.session_state.green_buffer) > 10:
                    sig = np.array(list(st.session_state.green_buffer))
                    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=sig_norm,
                        mode='lines',
                        line=dict(color='#4ade80', width=1.5),
                        name='Green Channel'
                    ))
                    fig.update_layout(
                        paper_bgcolor='#13131f',
                        plot_bgcolor='#13131f',
                        font=dict(family='Space Mono', color='#888899', size=10),
                        margin=dict(l=10, r=10, t=30, b=10),
                        title=dict(text='rPPG Signal', font=dict(size=11, color='#666680')),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=True, gridcolor='#1e1e2e', zeroline=False),
                        height=180,
                        showlegend=False,
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            time.sleep(1 / fs)

        cap.release()