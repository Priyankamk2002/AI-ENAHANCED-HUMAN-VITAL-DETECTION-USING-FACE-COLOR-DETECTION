import sys
import cv2
import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QFrame
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

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

def calculate_heart_rate(signal, fs):
    if len(signal) < fs * 5:
        return None
    signal = (signal - np.mean(signal)) / np.std(signal)
    filtered_signal = bandpass_filter(signal, 0.8, 2.5, fs)
    peaks, _ = find_peaks(filtered_signal, distance=fs//2, height=np.percentile(filtered_signal, 80)) 
    if len(peaks) < 2:
        return None
    peak_intervals = np.diff(peaks) / fs
    avg_interval = np.median(peak_intervals)
    bpm = int(60 / avg_interval)
    return bpm if 50 <= bpm <= 120 else None

def calculate_spo2(red_signal, green_signal):
    if len(red_signal) < 10 or len(green_signal) < 10:
        return None
    ac_red, dc_red = np.std(red_signal), np.mean(red_signal)
    ac_green, dc_green = np.std(green_signal), np.mean(green_signal)
    r_ratio = (ac_red / dc_red) / (ac_green / dc_green)
    spo2 = 104 - (20 * r_ratio)
    return max(90, min(100, int(spo2)))  

class HeartRateMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Heart Rate & SpO₂ Monitoring")
        self.setGeometry(100, 100, 1000, 500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_widget.setLayout(self.left_layout)

        self.webcam_label = QLabel("Press Start to Begin")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setFrameShape(QFrame.Box)
        self.left_layout.addWidget(self.webcam_label, stretch=3)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_monitoring)
        self.left_layout.addWidget(self.start_button, stretch=1)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.left_layout.addWidget(self.stop_button, stretch=1)
        self.stop_button.hide()

        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)

        self.info_label = QLabel("HR: N/A BPM | SpO₂: N/A %")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.info_label, stretch=1)

        self.plot_widget = pg.PlotWidget(title="Green Signal Intensity")
        self.right_layout.addWidget(self.plot_widget, stretch=3)
        self.green_curve = self.plot_widget.plot(pen='g')

        self.layout.addWidget(self.left_widget, stretch=1)
        self.layout.addWidget(self.right_widget, stretch=1)

        self.webcam = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.fs = 30
        self.green_signal_buffer = deque(maxlen=self.fs * 10)
        self.red_signal_buffer = deque(maxlen=self.fs * 10)

    def start_monitoring(self):
        self.start_button.hide()
        self.stop_button.show()
        if self.webcam is None:
            self.webcam = cv2.VideoCapture(0)
            self.timer.start(33)

    def stop_monitoring(self):
        self.timer.stop()
        if self.webcam:
            self.webcam.release()
        self.webcam = None
        self.webcam_label.setText("Press Start to Begin")
        self.info_label.setText("HR: N/A BPM | SpO₂: N/A %")
        self.green_signal_buffer.clear()
        self.red_signal_buffer.clear()
        self.green_curve.clear()
        self.start_button.show()
        self.stop_button.hide()

    def update_frame(self):
        ret, frame = self.webcam.read()
        if not ret:
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
            roi = extract_roi(frame, faces[0])
            if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                green_channel = roi[:, :, 1]
                red_channel = roi[:, :, 2]

                self.green_signal_buffer.append(np.mean(green_channel))
                self.red_signal_buffer.append(np.mean(red_channel))

                if len(self.green_signal_buffer) >= self.fs * 5:
                    heart_rate = calculate_heart_rate(list(self.green_signal_buffer), self.fs)
                    spo2 = calculate_spo2(list(self.red_signal_buffer), list(self.green_signal_buffer))

                    self.info_label.setText(f"HR: {heart_rate if heart_rate else 'N/A'} BPM | SpO₂: {spo2 if spo2 else 'N/A'} %")
                    self.green_curve.setData(list(self.green_signal_buffer))

        self.display_frame(frame)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qimg = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.webcam_label.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeartRateMonitor()
    window.show()
    sys.exit(app.exec_())