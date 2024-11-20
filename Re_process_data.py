
""" Download script for MIMIC-III data

In order to reproduce the results in the Sensors publication "Assessment of non-invasive blood pressure prediction from
PPG and rPPG signals using deep learning" the exact same data as used in the paper is downloaded. The record names are
provided in a text file. The scripts downloads those records, extract PPG and ABP data and performs peak detection on the
ABP (systolic and diastolic peaks of the ABP signals to generate systolic and diastolic blood pressure values as ground
truth) and PPG signals. ABP and PPG signals as well as the detected peaks are stored in .h5 files.

File: download_mimic_iii_records.py
Author: Dr.-Ing. Fabian Schrumpf
E-Mail: Fabian.Schrumpf@htwk-leipzig.de
Date created: 8/4/2021
Date last modified: 8/4/2021
"""

from itertools import compress
import datetime
from os.path import join, isdir, isfile
from os import chdir, mkdir, scandir, listdir 
from sys import argv
import argparse
import pandas as pd

import h5py
import numpy as np

import heartpy as hp
from scipy.signal import butter, filtfilt

old_dir = "D:\\downloadMIMICIII\\raw"
new_dir = "D:\\RPPG\\newraw"
chdir('D:\\RPPG')
RecordsFile = 'MIMIC-III_ppg_dataset_records.txt' #Record to be downloaded
ProcessResult = 'trans_result.csv'

# helper function to find minima between two macima
def find_minima(sig, pks, fs):  #if the window(signals between two peak) length < 1.5*fs, then lowest value in the windows will be considered diastolic
    min_pks = []
    for i in range(0,len(pks)):
        pks_curr = pks[i]
        if i == len(pks)-1:
            pks_next = len(sig)
        else:
            pks_next = pks[i+1]

        sig_win = sig[pks_curr:pks_next]
        if len(sig_win) < 1.5*fs:
            min_pks.append(np.argmin(sig_win) + pks_curr)
        else:
            min_pks.append(np.nan)

    return min_pks


def CreateWindows_fixbeat(beat_len, N_samp, overlap): #return start/end window in peak index array
    overlap = np.floor(beat_len*overlap)
    idx_start = np.floor(np.arange(0,N_samp-beat_len+1, overlap)).astype(int) 
    idx_stop = np.floor(idx_start + beat_len - 1).astype(int) 

    return idx_start, idx_stop




def download_mimic_iii_records(RecordsFile, OutputPath):
    # 4th order butterworth filter for PPG preprcessing
    b,a = butter(4,[0.5, 8], 'bandpass', fs=25)
    fs = 25
    SubjectDirs = scandir(old_dir)
    NumSubjects = sum(1 for x in SubjectDirs)  #count the subjects no. 
    SubjectDirs = scandir(old_dir)  #get the iterator again
    with open('completed.txt', 'r') as f: # read the processed samples list
        ProcessedList = f.read()
        ProcessedList = ProcessedList.split('\n')
        ProcessedList = ProcessedList[1:]
    for dirs in SubjectDirs: #process subject by subject under DataPath
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing ({dirs.name}): ', end='')
        DataFiles = [f for f in listdir(dirs) if isfile(join(dirs,f)) and f.endswith('.h5')] #get all sample files for this subject
        
        
        for file in DataFiles:
            # if file in ProcessedList:
            #     continue
            try:
                with h5py.File(join(dirs, file), "r") as f:
                    data = {}
                    for key in f.keys():
                        data[key] = np.array(f[key]).transpose()
            except TypeError:
                print("could not read file. Skipping.")
                continue
            ppg = data['val'][1, :]  #ppg data
            abp = data['val'][0, :] #abp data
 

            print(f'Processing record {file}')

            # with open('completed.txt', 'a') as f:
            #     f.write(file+'\n')

        
            ppg = filtfilt(b,a, ppg)
            abp = filtfilt(b,a, abp)
        
            try:
                abp_FidPoints = hp.process(abp, fs)
            except hp.exceptions.BadSignalWarning:
                df=pd.DataFrame([[file,'Heartpy could not process abp',len(abp)*5]],columns=['name','status','length'])
                df.to_csv(ProcessResult,mode='a',index=False,header=False)
                continue

            ValidPks = abp_FidPoints[0]['binary_peaklist'] #among all peaklist, 0: invalid, 1: valid
            abp_sys_pks = abp_FidPoints[0]['peaklist']  #systolic index in abp
            abp_sys_pks = list(compress(abp_sys_pks, ValidPks == 1)) #keep only valid(systolic) point
            abp_dia_pks = find_minima(abp, abp_sys_pks, fs) #diastolic index in abp
        
        

            try:
                ppg_FidPoints = hp.process(ppg, fs)
            except hp.exceptions.BadSignalWarning:
                df=pd.DataFrame([[file,'Heartpy could not process ppg',len(ppg)*5]],columns=['name','status','length'])
                df.to_csv(ProcessResult,mode='a',index=False,header=False)
                continue

            ValidPks = ppg_FidPoints[0]['binary_peaklist']
            ppg_pks = ppg_FidPoints[0]['peaklist']
            ppg_pks = list(compress(ppg_pks, ValidPks == 1))
            ppg_onset_pks = find_minima(ppg, ppg_pks, fs)

            # save ABP and PPG signals as well as detected peaks in a .h5 file

            SubjectName = file.split('_')[0]
            SubjectFolder = join(join(new_dir, SubjectName))
            if not isdir(SubjectFolder):
                mkdir(SubjectFolder)
    
            with h5py.File(join(SubjectFolder, file),'w') as f:
                signals = np.concatenate((abp[:,np.newaxis],ppg[:,np.newaxis]), axis=1)
                f.create_dataset('val', signals.shape, data=signals)
                f.create_dataset('nB2', (1,len(ppg_onset_pks)), data=ppg_onset_pks)
                f.create_dataset('nA2', (1,len(ppg_pks)), data=ppg_pks)
                f.create_dataset('nB3', (1,len(abp_dia_pks)), data=abp_dia_pks)
                f.create_dataset('nA3', (1,len(abp_sys_pks)), data=abp_sys_pks)
            df=pd.DataFrame([[file,'transfered successed',len(ppg)*5]],columns=['name','status','length'])
            df.to_csv(ProcessResult,mode='a',index=False,header=False)
 
    print('script finished')

if __name__ == '__main__':
    if len(argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=str,help='File containing the names of the records downloaded from the MIMIC-III DB')
        parser.add_argument('output', type=str, help='Folder for storing downloaded MIMIC-III records')

        args = parser.parse_args()

        RecordsFile = args.input
        OutputPath = args.output

        download_mimic_iii_records(RecordsFile, OutputPath)
    else:
        download_mimic_iii_records(RecordsFile, 'raw')
    