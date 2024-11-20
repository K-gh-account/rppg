# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:20:59 2024

@author: kenne
"""

import dlib
import cv2 as cv
import time
from queue import Queue
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# import heartpy as hp
# from itertools import compress

fps = 25
win_size = 7*fps

class CAM2FACE:
    def __init__(self,cm=True) -> None:
        # get face detector and 68 face landmark
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'data/shape_predictor_81_face_landmarks.dat')

        # get frontal camera of computer and get fps
        if cm:
            self.cam = cv.VideoCapture(0,cv.CAP_DSHOW)
            self.cam.set(cv.CAP_PROP_FPS, 25)
            if not self.cam.isOpened():
                print('ERROR:  Unable to open cam.  Verify that webcam is connected and try again.  Exiting.')
                self.cam.release()
                return
            self.fps = 25
        else:
            self.cam = cv.VideoCapture('test.mp4')
            if not self.cam.isOpened():
                print("ERROR: Unable to open video file.")
                self.cam.release()
                return
            self.fps = self.cam.get(cv.CAP_PROP_FPS)
            self.QUEUE_MAX = win_size + 3 * fps - 1
            # self.Queue_Sig_left = Queue(maxsize=self.QUEUE_MAX)
            # self.Queue_Sig_right = Queue(maxsize=self.QUEUE_MAX)
            self.Queue_Sig_fore = Queue(maxsize=self.QUEUE_MAX)
            self.Flag_Queue = False
            # self.Sig_left = None
            # self.Sig_right = None
            self.Sig_fore = None
            self.Frame_count = 0
            
        self.width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))


    def camera_process(self):
        Frame_count = 0
        frame_queue = Queue(maxsize = 300)
        video_length = win_size + 3 * fps 
        while True:
            self.ret, frame = self.cam.read()

            if not self.ret:
                self.Ongoing = False
                print("failed camera")
                break

            img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.detector(img_gray)
            if len(faces) == 1:

                Frame_count += 1
                frame_queue.put(frame)
                cv.imshow("Frame",frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                if Frame_count == video_length:
                    break
            else:
                Frame_count = 0
                frame_queue.queue.clear()
                continue
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
            
        videoWrite = cv.VideoWriter(r'test.mp4', fourcc, self.fps, (self.width,self.height))
        
        
        if Frame_count == video_length:
            for i in range(video_length):
                save_frame = frame_queue.get()
                videoWrite.write(save_frame)
            
        cv.destroyAllWindows()
            


    # Process: capture frame from camera in specific fps of the camera
    def capture_process(self):
        # Frame_count = 0
        SP = True
        ct = 0
        while True:
            self.ret, frame = self.cam.read()
            if not self.ret:
                print("video file error or too short.\n")
                break
            if ct == 100:
                pd.DataFrame(frame[0]).to_csv('frame1.csv')
            
            ct += 1

            if SP:
                SP = False
                continue
            self.roi_cal_process(frame)
            if self.Flag_Queue:
                break

    # Process: calculate roi from raw frame
    def roi_cal_process(self,frame):
        # check ROI exsistance
        ROI_fore = self.ROI(frame)
        if ROI_fore is not None:
            self.hist_fore = self.RGB_hist(ROI_fore)
            self.Queue_Sig_fore.put_nowait(self.Hist2Feature(self.hist_fore))
            if self.Queue_Sig_fore.full():
                self.Sig_fore = copy.copy(list(self.Queue_Sig_fore.queue))
                self.Flag_Queue = True
        else:
            self.hist_fore = None
            self.Flag_queue = False
            self.Queue_Sig_fore.queue.clear()


    # Get the markpoint of the faces
    def Marker(self, img):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.detector(img_gray)
        if len(faces) == 1:
            face = faces[0]
            landmarks = [[p.x, p.y] for p in self.predictor(img, face).parts()]

        try:
            return landmarks
        except:
            return None

    # filter the image to ensure better performance

    def preprocess(self, img):
        return cv.GaussianBlur(img, (5, 5), 0)

    # Draw the ROI the image
    # ROI: left cheek and right cheek

    def ROI(self, img):
        img = self.preprocess(img)
        landmark = self.Marker(img)
        if landmark is None:
            self.face_mask = img
            self.Flag_face = False
            return None

        forehead = [69, 70, 71, 80, 72, 25, 24, 23, 22, 21, 20, 19, 18]

        mask_fore = np.zeros(img.shape, np.uint8)
        try:
            self.Flag_face = True
            pts_fore = np.array([landmark[i]
                                 for i in forehead], np.int32).reshape((-1, 1, 2))
            mask_fore = cv.fillPoly(mask_fore, [pts_fore], (255, 255, 255))

            # Erode Kernel: 30
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 30))
            mask_fore = cv.erode(mask_fore, kernel=kernel, iterations=1)

            mask_display_fore = copy.copy(mask_fore)

            mask_display_fore[:, :, 2] = 0

            self.face_mask = cv.addWeighted(mask_display_fore, 0.25, img, 1, 0)

            ROI_fore = cv.bitwise_and(mask_fore, img)
            return ROI_fore
        
        except: #Exception as e:
            self.face_mask = img
            self.Flag_face = False
            return None
            #return None, None, None

    # Cal hist of roi

    def RGB_hist(self, roi):
        b_hist = cv.calcHist([roi], [0], None, [256], [0, 256])
        g_hist = cv.calcHist([roi], [1], None, [256], [0, 256])
        r_hist = cv.calcHist([roi], [2], None, [256], [0, 256])
        b_hist = np.reshape(b_hist, (256))
        g_hist = np.reshape(g_hist, (256))
        r_hist = np.reshape(r_hist, (256))
        b_hist[0] = 0
        g_hist[0] = 0
        r_hist[0] = 0
        r_hist = r_hist/np.sum(r_hist)
        g_hist = g_hist/np.sum(g_hist)
        b_hist = b_hist/np.sum(b_hist)
        return [r_hist, g_hist, b_hist]


    def Hist2Feature(self, hist):
        hist_r = hist[0]
        hist_g = hist[1]
        hist_b = hist[2]

        hist_r /= np.sum(hist_r)
        hist_g /= np.sum(hist_g)
        hist_b /= np.sum(hist_b)

        dens = np.arange(0, 256, 1)
        mean_r = dens.dot(hist_r)
        mean_g = dens.dot(hist_g)
        mean_b = dens.dot(hist_b)

        return [mean_r, mean_g, mean_b]

    # Deconstruction

    def __del__(self):
        self.Ongoing = False
        self.cam.release()
        cv.destroyAllWindows()

def max_window(data):
    window = 20
    for i in range(len(data)-window+1):
        sample = data[i:i+window]
        if np.argmax(sample) == 9:
            return i+9

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
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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

if __name__ == '__main__':
    # cam2video = CAM2FACE(cm=True)
    # cam2video.camera_process()
    # time.sleep(1)
    # cam2video.__del__()

    video2roi = CAM2FACE(cm=False)
    print("video fps: ", video2roi.cam.get(cv.CAP_PROP_FPS))
    video2roi.capture_process()
    
    # rppg_left = pos_process(video2roi.Sig_left)
    # rppg_right = pos_process(video2roi.Sig_right)
    rppg_fore = pos_process(video2roi.Sig_fore)
       
    # rppg_left = butter_bandpass_filter(rppg_left, 0.7, 4, fps)
    # rppg_right = butter_bandpass_filter(rppg_right, 0.7, 4, fps)
    rppg_fore = butter_bandpass_filter(rppg_fore, 0.7, 4, fps)
    
    # rppg_left_norm = (rppg_left - np.mean(rppg_left)) / np.std(rppg_left)
    # rppg_right_norm = (rppg_right - np.mean(rppg_right)) / np.std(rppg_right)
    rppg_fore_norm = (rppg_fore - np.mean(rppg_fore)) / np.std(rppg_fore)

    # start_index = np.argmax(rppg_fore_norm[:3*fps])
    # start_index = 0
    start_index = max_window(rppg_fore_norm[:50])
    rppg_fore_norm = rppg_fore_norm[start_index:start_index + win_size]
    
   
    plt.plot(rppg_fore_norm)
    plt.show()
    video2roi.__del__()