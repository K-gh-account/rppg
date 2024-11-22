
from itertools import compress
import datetime
from os.path import join, isdir, isfile
from os import chdir, mkdir
from sys import argv
import argparse
import pandas as pd

import h5py
import numpy as np
import wfdb
import heartpy as hp
from scipy.signal import butter, freqs, filtfilt

chdir('D:\\RPPG')
RecordsFile = 'MIMIC-III_ppg_dataset_records.txt' #Record to be downloaded
ProcessResult = 'process_result.csv'

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
    b,a = butter(5,[0.7, 4], 'bandpass', fs=25)

    # load record names from text file
    with open(RecordsFile, 'r') as f:  #read txt file include subject id/sample id
        RecordFiles = f.read()
        RecordFiles = RecordFiles.split("\n")  #remove \n in line
        RecordFiles = RecordFiles[:-1]  #remove the last record(blank record)
    
    with open('processed.txt', 'r') as f: # read the processed samples list
        ProcessedList = f.read()
        ProcessedList = ProcessedList.split('\n')
        ProcessedList = ProcessedList[1:]
    
   
    for file in RecordFiles:
        if file in ProcessedList:
            continue
        print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: Processing record {file}')
        # if len(file) < 15:
        #     print('skip single sample subject')
        #     continue

        # download record
        # with open('processed.txt', 'a') as f:
        #     f.write(file+'\n')
        try:
            record = wfdb.rdrecord(file.split('/')[1], pn_dir='mimic3wdb/' + file.split('_')[0])
        except:
            df=pd.DataFrame([[file,'Failed reading wfdb',0]],columns=['name','status','length'])
            df.to_csv(ProcessResult,mode='a',index=False,header=False)
            continue

        
        # check, if ABP and PLETH are present in the record. If not, continue with next record
        if 'PLETH' in record.sig_name:
            pleth_idx = record.sig_name.index('PLETH')  #index of ppg data in the signal data
            ppg = record.p_signal[:,pleth_idx] #draw the ppg data
            if len(ppg) < 3750:
                df=pd.DataFrame([[file,'too short ppg signal',len(ppg)]],columns=['name','status','length'])
                df.to_csv(ProcessResult,mode='a',index=False,header=False)
                continue
            fs = record.fs /  5
        else:
            df=pd.DataFrame([[file,'missing ppg signal',0]],columns=['name','status','length'])
            df.to_csv(ProcessResult,mode='a',index=False,header=False)
            continue
    
        if 'ABP' in record.sig_name:
            abp_idx = record.sig_name.index('ABP') #index of ABP data in the signal data
            abp = record.p_signal[:,abp_idx] #draw the ABP data
            if len(abp) < 3750:
                df=pd.DataFrame([[file,'too short abp signal',len(abp)]],columns=['name','status','length'])
                df.to_csv(ProcessResult,mode='a',index=False,header=False)
                continue
        else:
            df=pd.DataFrame([[file,'missing abp signal',0]],columns=['name','status','length'])
            df.to_csv(ProcessResult,mode='a',index=False,header=False)
            continue

        ppg = ppg[::5]
        abp = abp[::5]
        
        ppg = filtfilt(b,a, ppg)
        abp = filtfilt(b,a, abp)
        
        # detect systolic and diastolic peaks using heartpy
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
        SubjectName = file.split('/')[1]
        SubjectName = SubjectName.split('_')[0]
        SubjectFolder = join(join(OutputPath, SubjectName))
        if not isdir(SubjectFolder):
            mkdir(SubjectFolder)
    
        with h5py.File(join(SubjectFolder, file.split('/')[1] + ".h5"),'w') as f:
            signals = np.concatenate((abp[:,np.newaxis],ppg[:,np.newaxis]), axis=1)
            f.create_dataset('val', signals.shape, data=signals)
            f.create_dataset('nB2', (1,len(ppg_onset_pks)), data=ppg_onset_pks)
            f.create_dataset('nA2', (1,len(ppg_pks)), data=ppg_pks)
            f.create_dataset('nB3', (1,len(abp_dia_pks)), data=abp_dia_pks)
            f.create_dataset('nA3', (1,len(abp_sys_pks)), data=abp_sys_pks)
        df=pd.DataFrame([[file,'download successed',len(ppg)*5]],columns=['name','status','length'])
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
        download_mimic_iii_records(RecordsFile, 'new_raw')
    