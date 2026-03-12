# 💓 AI Enhanced Human Vital Detection Using Face Color Detection

A **non-contact, real-time vital sign monitoring system** that detects **Heart Rate (BPM)** and **Oxygen Saturation (SpO₂)** using just a webcam — no physical sensors needed!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📸 Output Preview

![Output](OUTPUT%20IMAGE.jpg)

---

## 🧠 Project Overview

This system captures live video of a person's face and processes the subtle **changes in skin color intensity** over time, which are directly linked to **blood volume pulse (BVP)**.

By isolating specific frequency components in the color signal, the program estimates:
- 💓 **Heart Rate (BPM)** — beats per minute
- 🫁 **SpO₂ (%)** — blood oxygen saturation level

---

## 🔍 How It Works

1. **Face Detection** — Detects face using OpenCV Haar Cascade algorithm
2. **ROI Extraction** — Focuses on cheek region for best pulse signal
3. **Color Signal Extraction** — Tracks green channel pixel intensity changes
4. **Signal Processing** — Bandpass filter (0.8–2.5 Hz) + FFT analysis
5. **Vital Estimation** — Dominant frequency → Heart Rate | Red/Green ratio → SpO₂
6. **Real-time Display** — Live HR, SpO₂ and signal graph

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV |
| Signal Processing | SciPy, NumPy |
| GUI (Desktop) | PyQt5, PyQtGraph |
| Web App | Streamlit, Plotly |
| Environment | VS Code / Jupyter Notebook |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Priyankamk2002/AI-ENAHANCED-HUMAN-VITAL-DETECTION-USING-FACE-COLOR-DETECTION.git
cd AI-ENAHANCED-HUMAN-VITAL-DETECTION-USING-FACE-COLOR-DETECTION
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Desktop App (PyQt5)
```bash
python main.py
```

### 4. Run Web App (Streamlit)
```bash
streamlit run app.py
```
Then open browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit>=1.32.0
opencv-python>=4.9.0
numpy>=1.26.0
scipy>=1.12.0
plotly>=5.19.0
pyqtgraph>=0.14.0
PyQt5>=5.15.0
```

---

## 💡 Tips for Best Results

- 💡 Use good lighting on your face
- 📏 Sit 40–60 cm from the camera
- ⏱ Wait 5–10 seconds for readings to stabilise
- 🧍 Keep your face centred and still
- 🚫 Avoid strong backlight or shadows

---

## 🔬 Motivation

Traditional devices like pulse oximeters require **physical contact**, which is inconvenient for remote monitoring. This project creates a **low-cost, camera-based solution** useful in:

- 🏥 Remote patient monitoring
- 🏃 Fitness tracking
- 💊 Preventive healthcare
- 🔬 Affective computing research

---

## 🚧 Future Scope

- 🩸 Blood Pressure estimation
- 😰 Stress Level detection
- 💓 Heart Rate Variability (HRV)
- 🌡️ Body Temperature estimation
- 📱 Mobile app integration

---

## 👩‍💻 Author

**Priyanka M K**
- GitHub: [@Priyankamk2002](https://github.com/Priyankamk2002)

---

## 📄 License

This project is licensed under the MIT License.

---

> ⭐ If you found this project helpful, please give it a star on GitHub!
