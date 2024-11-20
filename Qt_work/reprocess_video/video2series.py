import threading
import time
from queue import Queue

import cv2 as cv
import dlib
import matplotlib.pyplot as plt
import numpy as np
import copy
#import seaborn as sns
from scipy.signal import butter, filtfilt
#sns.set()
BufferSize = 1050
fps = 25
win_size = 40*fps

class VIDEO2FACE:
    def __init__(self, video) -> None:
        # get frontal camera of computer and get fps
        self.detector = dlib.get_frontal_face_detector()
        self.detector2 = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        self.predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')

        self.cam = cv.VideoCapture(video,cv.CAP_FFMPEG)
        self.cam.set(cv.CAP_PROP_BUFFERSIZE, 3)
        self.video = None
        if not self.cam.isOpened():
            print('ERROR:  Unable to open video file.  Verify that the path is correct and try again.  Exiting.')
            self.cam.release()
            return

        self.fps = fps
        self.width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.face_detected = None

        # Initialize Queue for camera capture
        self.Queue_Sig_fore = Queue(maxsize=BufferSize)
        self.Flag_Queue = False
        self.Flag_face = False  #flag that face is detected in the image
        self.Flag_Queue = False #flat that the sig_fore is successfully obtained
        self.face_mask = None
        self.hist_fore = None
        self.Sig_fore = None
        self.Sig_fore_out = None

    def process(self):
        self.video2signal()
        
    def video2signal(self):
        rtn, vid = self.cam.read()
        time.sleep(0.1)
        while True:
            rtn, vid = self.cam.read()
            #time.sleep(0.1)
            if not rtn:
                print("error recognize face")
                return False
            if vid is not None:
                if not self.roi_cal_process(vid):
                    return False
            if self.Flag_Queue:
                break
        print("completed read video file")
        self.cam.release()
        rppg_fore = pos_process(self.Sig_fore)
        rppg_fore = butter_bandpass_filter(rppg_fore, 0.3, 4, self.fps)
        #rppg_fore_norm = (rppg_fore - np.mean(rppg_fore)) / np.std(rppg_fore)
        rppg_fore_norm = rppg_fore
        start_index = max_window(rppg_fore_norm[:50])
        try:
            rppg_fore_norm = rppg_fore_norm[start_index:start_index + win_size]
        except:
            print("error process signal  ", start_index)
            return False
        self.sig_fore_out = copy.copy(rppg_fore_norm)
        return True


    # Process: calculate roi from raw frame
    def roi_cal_process(self, frame):
        # check ROI exsistance
        ROI_fore = self.ROI(frame)
        if ROI_fore is not None:
            self.hist_fore = self.RGB_hist(ROI_fore)
            self.Queue_Sig_fore.put_nowait(self.Hist2Feature(self.hist_fore))
            if self.Queue_Sig_fore.full():
                self.Sig_fore = copy.copy(list(self.Queue_Sig_fore.queue))
                self.Queue_Sig_fore.queue.clear()
                self.Flag_Queue = True
            return True
        else:
            self.hist_fore = None
            self.Flag_queue = False
            self.Queue_Sig_fore.queue.clear()
            return False

    # Get the markpoint of the faces

    def Marker(self, img):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.detector(img_gray)
        faces_len = len(faces)
        if faces_len >= 1:
            face = faces[0]
            cv.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 0, 0), thickness=1)
            self.face_detected = img
            landmarks = [[p.x, p.y] for p in self.predictor(img, face).parts()]
        else:
            print("face detection failure, try second way...")
            faces = self.detector2.detectMultiScale(img_gray,1.1,1)
            if faces_len > 0:
                face = faces[0]
                face = dlib.rectangle(face[0,0],face[0,1],face[0,0]+face[0,2],face[0,1]+face[0,3])
                cv.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 0, 0), thickness=1)
                self.face_detected = img
                landmarks = [[p.x, p.y] for p in self.predictor(img, face).parts()]
            else:
                print('way 2 failed also')
                return None
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
        img = self.preprocess(img)   #Gaussian Blur the Image
        landmark = self.Marker(img)  #Find the ROI, if success, the image include ROI area stored in self.face_detected
        if landmark is None:
            self.face_mask = img
            self.Flag_face = False
            return None
        forehead = [69, 70, 71, 80, 72, 25, 24, 23, 22, 21, 20, 19, 18]
        mask_fore = np.zeros(img.shape, np.uint8)
        try:
            self.Flag_face = True
            pts_fore = np.array([landmark[i] for i in forehead], np.int32).reshape((-1, 1, 2))
            mask_fore = cv.fillPoly(mask_fore, [pts_fore], (255, 255, 255))

            # Erode Kernel: 30
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 30))
            mask_fore = cv.erode(mask_fore, kernel=kernel, iterations=1)
            mask_display_fore = copy.copy(mask_fore)
            mask_display_fore[:, :, 2] = 0
            self.face_mask = cv.addWeighted(mask_display_fore, 0.25, img, 1, 0)
            ROI_fore = cv.bitwise_and(mask_fore, img)
            return ROI_fore

        except:  # Exception as e:
            self.face_mask = img
            self.Flag_face = False
            return None


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
        try:
            r_hist = r_hist / np.sum(r_hist)
            g_hist = g_hist / np.sum(g_hist)
            b_hist = b_hist / np.sum(b_hist)
            return [r_hist, g_hist, b_hist]
        except:
            print(np.sum(r_hist), np.sum(g_hist), np.sum(b_hist))

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

def max_window(data):
    window = 20
    for i in range(len(data) - window + 1):
        sample = data[i:i + window]
        if np.argmax(sample) == 9:
            return i + 9
    return i+np.argmax(sample)

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

def pos_process(buffer):
    # Temporal normalization
    Cn = buffer / np.mean(buffer, axis=0)

    # Project to plane orthogonal to skin-tone
    S = np.array([[0, 1, -1], [-2, 1, 1]])
    P = np.dot(Cn, S.T)

    # Alpha tuning
    Q = P[:, 0] + ((np.std(P[:, 0]) / np.std(P[:, 1])) * P[:, 1])

    return Q


