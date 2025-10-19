# 💓 Human Vital Detection using Facial Skin Color Variation

This project demonstrates a **non-contact method** to estimate **Heart Rate (BPM)** and **Oxygen Saturation (SpO₂)** using a **webcam** or **video input**.  
It leverages **Computer Vision** and **Signal Processing** techniques to analyze **facial skin color variations** caused by blood flow beneath the skin — allowing real-time vital sign monitoring **without any physical sensors**.

---

## 🧠 Project Overview

The system captures live video of a person’s face and processes the subtle **changes in skin color intensity** over time, which are directly linked to **blood volume pulse (BVP)**.  
By isolating specific frequency components in the color signal, the program estimates the user’s **heart rate**, and by analyzing **color ratio features**, it estimates **SpO₂ (oxygen saturation)**.

### 👨‍⚕️ Vital Parameters Detected
- 💓 **Heart Rate (BPM)** — beats per minute  
- 🫁 **Oxygen Saturation (SpO₂%)** — blood oxygen level

> **Future Scope:** Integration of Blood Pressure, Stress Level, Heart Rate Variability (HRV), and Body Temperature.

---

## 🔬 Motivation

Traditional medical devices like pulse oximeters and ECG sensors require **physical contact**, which can be inconvenient in telehealth or remote monitoring scenarios.  
This project aims to create a **low-cost, camera-based solution** that enables **contactless, real-time health monitoring**, especially useful in:
- Remote patient monitoring  
- Fitness tracking  
- Preventive healthcare  
- Research in affective computing  

---

## 🔍 How It Works

1. **Face Detection**  
   Detects the face in each video frame using **OpenCV’s Viola–Jones (Haar Cascade)** algorithm.

2. **Region of Interest (ROI) Extraction**  
   Focuses on skin regions like **cheeks** or **forehead**, which are ideal for detecting pulse signals due to good blood flow visibility.

3. **Color Signal Extraction**  
   Tracks pixel intensity changes in the **green channel**, as it carries the strongest blood pulse signal compared to red or blue.

4. **Signal Processing**  
   - A **Bandpass Filter** (0.7–4 Hz) removes noise and isolates frequencies corresponding to normal heart rate range.  
   - The **Fast Fourier Transform (FFT)** converts this signal into the frequency domain.

5. **Vital Estimation**  
   - The **dominant frequency peak** in the FFT is converted into **Heart Rate (BPM)**.  
   - The **ratio of red and green signals** provides an approximation of **SpO₂** levels.

6. **Result Display**  
   - Outputs HR and SpO₂ in real time.  
   - Plots of raw and filtered signals are generated for visualization and validation.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Programming Language** | Python |
| **Libraries Used** | OpenCV, NumPy, SciPy, Matplotlib, PyQt5 |
| **Interface / GUI** | Streamlit or PyQt5 |
| **Development Environment** | Jupyter Notebook / VS Code |

---


