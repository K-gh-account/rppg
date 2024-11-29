# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:57:28 2024
change the filter to (0.3,4)
@author: kenne
"""

from os.path import join, isfile, splitext, isdir
from os import chdir, mkdir, listdir
import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import tensorflow as tf
import readchar
from video2series import VIDEO2FACE
import matplotlib.pyplot as plt
import sys

DataLength = 175
chdir('D:\\Qt_work\\reprocess_video')

def Load_video_data(DataPath, VideoPath, OutputPath):
    DataDir = listdir(DataPath)

    for file in DataDir:
        dataName = splitext(file)[0]
        infoList = dataName.split('_')
        videoName = infoList[0]+'.mp4'
        videoName = join(VideoPath,videoName)
        print(videoName)
        SBP_char = infoList[1]  #string type
        DBP_char = infoList[2]  #string type

        processor = VIDEO2FACE(videoName)
        processor.process()
        if processor.sig_fore_out is None:
            del(processor)
            continue

        plt.plot(processor.sig_fore_out)

        plt.show()

        dataName = infoList[0] + '_' + SBP_char + '_' + DBP_char + '.csv'
        dataName = join(OutputPath, dataName)
        pd.DataFrame(processor.sig_fore_out).to_csv(dataName)
        del(processor)


        
if __name__ == "__main__":
    Load_video_data('rppg_data', 'rppg_video', 'new_data')
    print('Completed')
