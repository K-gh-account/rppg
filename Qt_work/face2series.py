import threading
import time
from queue import Queue

import cv2 as cv
import dlib
import matplotlib.pyplot as plt
import numpy as np
import copy
import seaborn as sns

sns.set()
BufferSize = 1050

class VIDEO2FACE:
    def __init__(self) -> None:
        # get frontal camera of computer and get fps
        self.detector = dlib.get_frontal_face_detector()
        self.detector2 = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        self.predictor = dlib.shape_predictor('data/shape_predictor_81_face_landmarks.dat')
        #self.cam = cv.VideoCapture(0,cv.CAP_DSHOW)
        self.cam = cv.VideoCapture(0)
        self.video = None
        self.cam.set(cv.CAP_PROP_FPS, 25)

        self.cam.set(3,1280)
        self.cam.set(4,720)
        self.cam.set(6, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        if not self.cam.isOpened():
            print('ERROR:  Unable to open webcam.  Verify that webcam is connected and try again.  Exiting.')
            self.cam.release()
            return
        #print(self.cam.get(cv.CAP_PROP_FPS))
        self.fps = 25
        # self.cam.set(cv.CAP_PROP_FPS, self.fps)
        self.width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        print(self.width,self.height)
        self.face_detected = None

        # Initialize Queue for camera capture
        self.Queue_rawframe = Queue(maxsize=BufferSize+1)
        self.Queue_saveframe = Queue(maxsize=BufferSize+1)

        self.Queue_Sig_fore = Queue(maxsize=BufferSize)
        self.Flag_Queue = False

        self.Ongoing = False
        self.Flag_face = False
        self.Flag_Queue = False

        self.frame_display = None
        self.face_mask = None

        self.Sig_fore = None
        self.Sig_fore_out = None

    # Initialize process and start
    def PROCESS_start(self):
        self.Ongoing = True
        self.capture_process_ = threading.Thread(target=self.capture_process)
        self.capture_process_.start()

    # Process: capture frame from camera in specific fps of the camera
    def capture_process(self):
        while self.Ongoing:
            # time.sleep(0.02)
            # get frame
            self.ret, frame = self.cam.read()

            if not self.ret:
                self.Ongoing = False
                break

            # check if rawframe queue is full, if true then clear the last data
            if self.Queue_rawframe.full():
                print("warning: raw video queue full")
                self.Queue_rawframe.get_nowait()

            try:
                self.Queue_rawframe.put_nowait(frame)
            except Exception as e:
                print("warning: save raw video fail")
                pass

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
        #elif faces_len > 1:
        #    min_cd = 1000000
        #    i_store = 0
        #    for i,d in faces:
        #        c_d = np.squar(d.left()+((d.right()-d.left())/2) - self.width/2) + np.squar(d.top()+((d.bottom()-d.top())/2 - self.height/2))
        #        if c_d < min_cd:
        #            i_store = i
        #            min_cd = c_d
        #    face = faces[i_store]
        #    cv.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 0, 0), thickness=1)
        #    self.face_detected = img
        #    landmarks = [[p.x, p.y] for p in self.predictor(img, face).parts()]
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

    def __del__(self):
        self.Ongoing = False
        self.cam.release()
        #cv.destroyAllWindows()


if __name__ == '__main__':
    cam2roi = CAM2FACE()
    cam2roi.PROCESS_start()

    time.sleep(1)

    cam2roi.__del__()
