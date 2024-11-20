# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:34:40 2024

@author: kenne
"""

from os.path import join, expanduser, isfile, splitext
from os import listdir, scandir, chdir
from random import shuffle
from sys import argv
import datetime
import argparse

import h5py
import numpy as np
from scipy.signal import butter, freqs, filtfilt
from sklearn.covariance import EllipticEnvelope
from matplotlib import pyplot as plt

chdir('D:\\RPPG')

def display_MIMIC_dataset(DataPath):

    SubjectDirs = scandir(DataPath)
    NumSubjects = sum(1 for x in SubjectDirs)  #count the subjects no. 
    SubjectDirs = scandir(DataPath)  #get the iterator again

    fs = 25

    # 4th order butterworth filter for PPG preprcessing
    b,a = butter(4,[0.5, 8], 'bandpass', fs=fs)

    # loop over all subjects and their files in the source folder
    subject_draw=True
    for idx, dirs in enumerate(SubjectDirs): #process subject by subject under DataPath
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing subject {idx+1} of {NumSubjects} ({dirs.name}): ', end='')
        if not subject_draw:
            break
        subject_draw = False
        #DataFiles = [f for f in listdir(join(DataPath,dirs)) if isfile(join(DataPath, dirs,f)) and f.endswith('.h5')] 
        DataFiles = [f for f in listdir(dirs) if isfile(join(dirs,f)) and f.endswith('.h5')] #get all sample files for this subject
        
        ct = 0
        for file in DataFiles:
            try:
                with h5py.File(join(dirs, file), "r") as f:
                    data = {}
                    for key in f.keys():
                        data[key] = np.array(f[key]).transpose()
            except TypeError:
                print("could not read file. Skipping.")

            ABP = data['val'][0, :] #abp data
            PPG = data['val'][1, :]
            
            #plt.plot(ABP[:500],'r-')
            plt.plot(PPG[:250],'g--')
            plt.show()
            
            ct = ct + 1
            if ct == 10:
                break
            


    return 0

if __name__ == "__main__":
    np.random.seed(seed=42)
    if len(argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('datapath', type=str,
                        help="Path containing data records downloaded from the MIMIC-III database")

        args = parser.parse_args()

        DataPath = args.datapath


        display_MIMIC_dataset(DataPath)
    else:
        display_MIMIC_dataset('.\\raw')