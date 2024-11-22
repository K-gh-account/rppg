

import sys
from main_rppg_window import Ui_MainWindow
import os
import shutil
import pandas as pd
from scipy.fft import rfft, rfftfreq
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import time
import copy
import pyqtgraph as pg

from obspy.signal.detrend import spline
from scipy import signal
import numpy as np
import cv2 as cv
from face2series import VIDEO2FACE
from queue import Queue
from scipy.signal import butter, filtfilt

fps = 25
win_size = 40*fps
BufferSize = 1050


class mainwin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(mainwin, self).__init__(parent)
        self.setupUi(self)

        self.Hist_fore = pg.PlotWidget(self)
        self.Hist_fore.setYRange(min=-3, max=3, padding=0.2)
        self.Hist_fore_signal = self.Hist_fore.plot()
        self.signallayout.addWidget(self.Hist_fore)
        self.heartRate = '60'

        self.Face.setScaledContents(True)
        self.processor = VIDEO2FACE()
        self.processor.PROCESS_start()

        self.TIMER_Frame = QTimer()
        self.TIMER_Frame.setInterval(25)
        self.TIMER_Frame.start()

        self.RECORDING = False

        self.slot_init()
        self.stop.setEnabled(False)
        self.submit.setEnabled(False)
        self.redo.setEnabled(False)
        self.next.setEnabled(False)
        self.SBP.setEnabled(False)
        self.DBP.setEnabled(False)
        self.SBP.setValidator(QIntValidator())
        self.DBP.setValidator(QIntValidator())

    def slot_init(self):
        self.TIMER_Frame.timeout.connect(self.DisplayImage)
        self.start.clicked.connect(self.RecordVideo)
        self.stop.clicked.connect(self.SaveVideo)
        self.quit.clicked.connect(self.QuitApp)
        self.submit.clicked.connect(self.SubmitData)
        self.redo.clicked.connect(self.ReTake)
        self.next.clicked.connect(self.NextOne)

    def Face2Signal(self):
        self.processor.Flag_Queue = False
        self.processor.Sig_fore = None
        vd = cv.VideoCapture("test.mp4",cv.CAP_FFMPEG)
        vd.set(cv.CAP_PROP_BUFFERSIZE, 3)
        rtn, vid = vd.read()
        time.sleep(0.1)
        while True:
            rtn, vid = vd.read()
            #time.sleep(0.1)
            if not rtn:
                print("error recognize face")
                return False
            if vid is not None:
                if not self.processor.roi_cal_process(vid):
                    return False
                img = cv.cvtColor(self.processor.face_detected, cv.COLOR_BGR2RGB)
                qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
                self.Face.setPixmap(QPixmap.fromImage(qimg))
                QApplication.processEvents()
            if self.processor.Flag_Queue:
                break

        vd.release()
        rppg_fore = pos_process(self.processor.Sig_fore)
        rppg_fore = butter_bandpass_filter(rppg_fore, 0.3, 4, self.processor.fps)
        rppg_fore_norm = (rppg_fore - np.mean(rppg_fore)) / np.std(rppg_fore)
        start_index = max_window(rppg_fore_norm[:50])
        try:
            rppg_fore_norm = rppg_fore_norm[start_index:start_index + win_size]
        except:
            print("error process signal  ", start_index)
            return False
        self.Hist_fore_signal.setData(rppg_fore_norm,pen=(255,0,0))
        self.processor.sig_fore_out = copy.copy(rppg_fore_norm)
        HR_rate = findHR(rppg_fore_norm)
        self.heartRate = HR_rate
        return True

    def SubmitData(self):
        SBP = self.SBP.text()
        DBP = self.DBP.text()

        if SBP == '' or DBP == '':
            reply = QMessageBox.about(self, "警告", "请输入合理的收缩压和舒张压")
            return False

        SBP = int(SBP)
        DBP = int(DBP)

        if (40 < SBP, DBP < 250) != (True,True) or SBP <= DBP:
            reply = QMessageBox.about(self, "警告", "请输入合理的收缩压和舒张压")
            return False
        name = str(int(time.time()))
        dataName = name + '_' + str(SBP) + '_' + str(DBP) + '.csv'
        videoName = name + '.mp4'

        dataName = os.path.join("rppg_data", dataName)
        videoName = os.path.join("rppg_video", videoName)
        pd.DataFrame(self.processor.sig_fore_out).to_csv(dataName)
        shutil.copyfile("test.mp4",videoName)
        self.status.setText("数据提交成功，请继续下一个，或结束程序")
        self.SBP.clear()
        self.DBP.clear()
        self.submit.setEnabled(False)
        self.next.setEnabled(True)
        self.start.setEnabled(False)
        self.redo.setEnabled(False)
        self.SBP.setEnabled(False)
        self.DBP.setEnabled(False)
        self.Hist_fore_signal.setData(np.array([]), pen=(255, 0, 0))

    def QuitApp(self):
        reply = QMessageBox.question(self, '提示',
                                     "是否要关闭所有窗口?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.processor.Ongoing = False
            sys.exit(app.exec_())
        else:
            return

    def NextOne(self):
        self.status.setText("重置摄像头，稍等。。。")
        self.processor.cam = cv.VideoCapture(0)
        self.processor.cam.set(cv.CAP_PROP_FPS, 25)
        self.processor.cam.set(3,1280)
        self.processor.cam.set(4,720)
        self.processor.cam.set(6, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.processor.Queue_saveframe.queue.clear()
        self.processor.Queue_rawframe.queue.clear()
        self.processor.Queue_Sig_fore.queue.clear()
        self.processor.Flag_face = False
        self.processor.Flag_Queue = False
        self.processor.frame_display = None
        self.processor.face_mask = None
        self.processor.Sig_fore = None
        self.processor.Sig_fore_out = None
        self.start.setEnabled(True)
        self.redo.setEnabled(False)
        self.status.setText("空闲中。。。")
        self.Hist_fore_signal.setData(np.array([]), pen=(255, 0, 0))
        self.processor.PROCESS_start()

    def ReTake(self):
        reply = QMessageBox.question(self, '提示',
                                     "重做将抛弃当前视频，是否继续?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.processor.cam = cv.VideoCapture(0)
        self.processor.cam.set(cv.CAP_PROP_FPS, 25)
        self.processor.cam.set(3,1280)
        self.processor.cam.set(4,720)
        self.processor.cam.set(6, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.processor.Queue_saveframe.queue.clear()
        self.processor.Queue_rawframe.queue.clear()
        self.processor.Queue_Sig_fore.queue.clear()
        self.processor.Flag_face = False
        self.processor.Flag_Queue = False
        self.processor.frame_display = None
        self.processor.face_mask = None
        self.processor.Sig_fore = None
        self.processor.Sig_fore_out = None
        self.SBP.clear()
        self.DBP.clear()
        self.submit.setEnabled(False)
        self.next.setEnabled(False)
        self.start.setEnabled(True)
        self.redo.setEnabled(False)
        self.SBP.setEnabled(False)
        self.DBP.setEnabled(False)
        self.status.setText("空闲中。。。")
        self.Hist_fore_signal.setData(np.array([]), pen=(255, 0, 0))
        self.processor.PROCESS_start()

    def RecordVideo(self):
        self.RECORDING = True
        self.processor.Queue_saveframe.queue.clear()

        self.start.setEnabled(False)
        self.status.setText("视频采集中。。。")
        return

    def SaveVideo(self):

        self.RECORDING = False
        self.processor.Ongoing = False
        self.stop.setEnabled(False)
        self.processor.cam.release()
        if self.processor.Queue_saveframe.qsize() <= BufferSize:
            self.status.setText("录制视频不足三十秒，请重做一次")
            QApplication.processEvents()
        else:
            self.status.setText("正在审核视频，并做信号处理，请稍等。。。")
            QApplication.processEvents()
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            videoWrite = cv.VideoWriter(r'test.mp4', fourcc, self.processor.fps, (self.processor.width, self.processor.height))
            while not self.processor.Queue_rawframe.empty():
                self.processor.Queue_saveframe.get_nowait()
                self.processor.Queue_saveframe.put_nowait(self.processor.Queue_rawframe.get_nowait())

            while not self.processor.Queue_saveframe.empty():
                currentFrame = self.processor.Queue_saveframe.get_nowait()
                videoWrite.write(currentFrame)

            time.sleep(2)
            videoWrite.release()
            if self.Face2Signal():
                self.status.setText("处理完成，心跳:"+self.heartRate+"   输入血压后提交，或重做。")
                self.SBP.setEnabled(True)
                self.DBP.setEnabled(True)
                self.submit.setEnabled(True)
                self.redo.setEnabled(True)
            else:
                self.status.setText("抱歉，图像中部分图片有问题，请重做。")
                self.redo.setEnabled(True)

    def DisplayImage(self):
        if self.processor.Queue_rawframe.empty():
            return

        image = self.processor.Queue_rawframe.get_nowait()

        if image is not None:
            img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            qimg = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            self.Face.setPixmap(QPixmap.fromImage(qimg))
            if self.RECORDING:
                if self.processor.Queue_saveframe.full():
                    self.processor.Queue_saveframe.get_nowait()
                    self.stop.setEnabled(True)
                    self.status.setText("视频时长已满足，按结束采集进行处理。")
                    #QApplication.processEvents()
                self.processor.Queue_saveframe.put_nowait(image)

    # Creates the specified Butterworth filter and applies it.
    def butterworth_filter(self, data, low, high, sample_rate, order=11):
        nyquist_rate = sample_rate * 0.5
        low /= nyquist_rate
        high /= nyquist_rate
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, data)

def to_frequency_domain(signal):
    # 计算FFT
    n = len(signal)
    freqs = rfftfreq(n, d=1.0/25)
    spectrum = rfft(signal)
    return freqs, spectrum

def findHR(signal):
    freqs, spectrum = to_frequency_domain(signal)
    amplitudes = np.abs(spectrum)
    sorted_indices = np.argsort(amplitudes)[::-1]
    freqs = freqs[sorted_indices]
    freqs = freqs[:6]
    for i in range(6):
        print(f"Rank {i+1}: Frequency {freqs[i]:.2f} Hz, Amplitude {amplitudes[sorted_indices[i]]:.2f}")
    HR = 0
    for freq in freqs:
        if freq>=1 and freq<2:
            return str(np.floor(freq*60).astype(int))
        if freq >= 2:
            if HR == 0:
                HR=freq
    if HR > 0:
        return str(np.floor(HR*60).astype(int))
    else:
        return str(np.floor(freqs[0]*60).astype(int))


def max_window(data):
    window = 20
    for i in range(len(data) - window + 1):
        sample = data[i:i + window]
        if np.argmax(sample) == 9:
            return i + 9


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
        face_roi = frame[y:y + h, x:x + w]

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

if __name__ == "__main__":
    os.chdir('D:\\Qt_work')
    app = QApplication(sys.argv)
    ui = mainwin()
    ui.show()
    sys.exit(app.exec_())
