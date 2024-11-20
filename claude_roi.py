# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:46:41 2024

@author: kenne
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt
import csv
import os

os.chdir('D:\\RPPG')


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
        
        # Define ROIs
        forehead_roi = frame[y:y+int(h*0.3), x+int(w*0.3):x+int(w*0.7)]
        left_cheek_roi = frame[y+int(h*0.4):y+int(h*0.7), x:x+int(w*0.3)]
        right_cheek_roi = frame[y+int(h*0.4):y+int(h*0.7), x+int(w*0.7):x+w]
        
        # Calculate mean RGB for each ROI
        forehead_rgb = np.mean(forehead_roi, axis=(0, 1))
        left_cheek_rgb = np.mean(left_cheek_roi, axis=(0, 1))
        right_cheek_rgb = np.mean(right_cheek_roi, axis=(0, 1))
        
        # Draw ROIs on the frame
        cv2.rectangle(frame, (x+int(w*0.3), y), (x+int(w*0.7), y+int(h*0.3)), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y+int(h*0.4)), (x+int(w*0.3), y+int(h*0.7)), (0, 255, 0), 2)
        cv2.rectangle(frame, (x+int(w*0.7), y+int(h*0.4)), (x+w, y+int(h*0.7)), (0, 255, 0), 2)
        
        return forehead_rgb, left_cheek_rgb, right_cheek_rgb
    else:
        return None, None, None

def pos_process(buffer):
    # Temporal normalization
    Cn = buffer / np.mean(buffer, axis=0)
    
    # Project to plane orthogonal to skin-tone
    S = np.array([[0, 1, -1], [-2, 1, 1]])
    P = np.dot(Cn, S.T)
    
    # Alpha tuning
    Q = P[:, 0] + ((np.std(P[:, 0]) / np.std(P[:, 1])) * P[:, 1])
    
    return Q

def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Time', 'Signal'])  # Header
        for i, value in enumerate(data):
            csv_writer.writerow([i / 25, value])  # Assuming 25 fps

def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap.set(cv2.CAP_PROP_FPS, 25)
    fps = 25
    buffer_size = fps * 10  # 10 seconds buffer
    
    rppg_buffers = {
        'forehead': np.zeros((buffer_size, 3)),
        'left_cheek': np.zeros((buffer_size, 3)),
        'right_cheek': np.zeros((buffer_size, 3))
    }
    
    signal_data = {
        'forehead': [],
        'left_cheek': [],
        'right_cheek': []
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        forehead_rppg, left_cheek_rppg, right_cheek_rppg = get_rppg_pos(frame, face_cascade)
        
        if forehead_rppg is not None:
            for roi, rppg in zip(['forehead', 'left_cheek', 'right_cheek'], 
                                 [forehead_rppg, left_cheek_rppg, right_cheek_rppg]):
                rppg_buffers[roi] = np.roll(rppg_buffers[roi], -1, axis=0)
                rppg_buffers[roi][-1] = rppg
                
                # Apply POS algorithm
                pos_signal = pos_process(rppg_buffers[roi])
                
                # Apply bandpass filter
                filtered_signal = butter_bandpass_filter(pos_signal, 0.7, 4, fps)
                
                # Normalize the signal
                normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
                
                # Store the signal data
                signal_data[roi].append(normalized_signal[-1])
                
                # Plot the signal
                signal_display = cv2.resize(np.repeat(normalized_signal[-300:, np.newaxis], 3, axis=1), (300, 100))
                cv2.imshow(f'rPPG Signal - {roi}', signal_display)
        
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save signal data to CSV files
    save_to_csv('forehead.csv', signal_data['forehead'])
    save_to_csv('leftface.csv', signal_data['left_cheek'])
    save_to_csv('rightface.csv', signal_data['right_cheek'])
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()