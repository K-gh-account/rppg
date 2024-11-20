# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:22:38 2024

@author: kenne
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def get_rppg_pos(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        mean_rgb = np.mean(face_roi, axis=(0, 1))
        return mean_rgb
    else:
        return None

def pos_process(buffer):
    # Temporal normalization
    Cn = buffer / np.mean(buffer, axis=0)
    
    # Project to plane orthogonal to skin-tone
    S = np.array([[0, 1, -1], [-2, 1, 1]])
    P = np.dot(Cn, S.T)
    
    # Alpha tuning
    Q = P[:, 0] + ((np.std(P[:, 0]) / np.std(P[:, 1])) * P[:, 1])
    
    return Q

def main():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    face_cascade = cv2.CascadeClassifier(cv2.haarcascades + 'haarcascade_frontalface_default.xml')
    cap.set(cv2.CAP_PROP_FPS, 25)
    fps = 25
    buffer_size = fps * 10  # 10 seconds buffer
    rppg_buffer = np.zeros((buffer_size, 3))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rppg = get_rppg_pos(frame, face_cascade)
        
        if rppg is not None:
            rppg_buffer = np.roll(rppg_buffer, -1, axis=0)
            rppg_buffer[-1] = rppg
            
            # Apply POS algorithm
            pos_signal = pos_process(rppg_buffer)
            
            # Apply bandpass filter
            filtered_signal = butter_bandpass_filter(pos_signal, 0.7, 4, fps)
            
            # Normalize the signal
            normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
            
            # Plot the signal (you can modify this part to display it as you prefer)
            cv2.imshow('rPPG Signal', cv2.resize(np.repeat(normalized_signal[-300:, np.newaxis], 3, axis=1), (300, 100)))
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()