# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:57:28 2024

@author: kenne
"""

from os.path import join, isfile, splitext, isdir
from os import chdir, mkdir, listdir
import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import keyboard

DataLength = 175
chdir('D:\\Qt_work')

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def rppg_train_hdf2tfrecord(h5_file, tfrecord_path, idx, modus="train", weights_SBP=None, weights_DBP=None):
  
    with h5py.File(h5_file, 'r') as f:
        # load ppg and BP data as well as the subject numbers the samples belong to
        ppg_h5 = f.get('/ppg')
        BP = f.get('/label')
        subject_idx = f.get('/subject_idx')

        for sf in range(10):
            output_filename = join(tfrecord_path, 'RETRAIN_RPPG_'+str(sf)+'_'+modus+'.tfrecord')
            writer = tf.io.TFRecordWriter(output_filename)
            np.random.shuffle(idx)
            print('Now processing '+ modus+ ' '+str(sf))
            for i in idx:
                rppg = np.array(ppg_h5[i,:])
                if weights_SBP is not None and weights_DBP is not None:
                    weight_SBP = weights_SBP[i]
                    weight_DBP = weights_DBP[i]
                else:
                    weight_SBP = 1
                    weight_DBP = 1

                target = np.array(BP[i,:], dtype=np.float32)
                sub_idx = np.array(subject_idx[i])

                # create a dictionary containing the serialized data
                data = \
                    {'ppg': _float_feature(rppg.tolist()),
                     'label': _float_feature(target.tolist()),
                     'subject_idx': _float_feature(sub_idx.tolist()),
                     'weight_SBP': _float_feature([weight_SBP]),
                     'weight_DBP': _float_feature([weight_DBP]),
                     'Nsamples': _float_feature([idx.shape[0]])}

                # write data to the .tfrecord target file
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()

                writer.write(serialized)

            writer.close()


def rppg_hdf2tfrecord(h5_file, tfrecord_path, modus="train", weights_SBP=None, weights_DBP=None):
    
    with h5py.File(h5_file, 'r') as f:
        # load ppg and BP data as well as the subject numbers the samples belong to
        ppg_h5 = f.get('/ppg')
        BP = f.get('/label')
        subject_idx = f.get('/subject_idx')
        output_filename = join(tfrecord_path, 'RETRAIN_RPPG_'+modus+'.tfrecord')


        writer = tf.io.TFRecordWriter(output_filename)
        idx = np.array(range(ppg_h5.shape[0]))
        np.random.shuffle(idx)
        ct = 0
        for i in idx:
            print('Now processing '+ modus+ ' '+str(ct))
            ct = ct + 1
            rppg = np.array(ppg_h5[i,:])
            if weights_SBP is not None and weights_DBP is not None:
                weight_SBP = weights_SBP[i]
                weight_DBP = weights_DBP[i]
            else:
                weight_SBP = 1
                weight_DBP = 1

            target = np.array(BP[i,:], dtype=np.float32)
            sub_idx = np.array(subject_idx[i])

            # create a dictionary containing the serialized data
            data = \
                {'ppg': _float_feature(rppg.tolist()),
                 'label': _float_feature(target.tolist()),
                 'subject_idx': _float_feature(sub_idx.tolist()),
                 'weight_SBP': _float_feature([weight_SBP]),
                 'weight_DBP': _float_feature([weight_DBP]),
                 'Nsamples': _float_feature([idx.shape[0]])}

            # write data to the .tfrecord target file
            feature = tf.train.Features(feature=data)
            example = tf.train.Example(features=feature)
            serialized = example.SerializeToString()

            writer.write(serialized)

        writer.close()

    

def prepare_retrain_dataset(DataPath, TrainFile, TestFile, ValFile):
    if not isfile(TrainFile):
        with h5py.File(TrainFile, "a") as f:  #create the h5 file
            f.create_dataset('ppg', (0,DataLength), maxshape=(None,DataLength), chunks=(100, DataLength))
            f.create_dataset('label', (0,2), maxshape=(None,2), dtype=int, chunks=(100,2))
            f.create_dataset('subject_idx', (0,1), maxshape=(None,1), dtype=int, chunks=(100,1))
    if not isfile(TestFile):
        with h5py.File(TestFile, "a") as f:  #create the h5 file
            f.create_dataset('ppg', (0,DataLength), maxshape=(None,DataLength), chunks=(100, DataLength))
            f.create_dataset('label', (0,2), maxshape=(None,2), dtype=int, chunks=(100,2))
            f.create_dataset('subject_idx', (0,1), maxshape=(None,1), dtype=int, chunks=(100,1))            
    if not isfile(ValFile):
        with h5py.File(ValFile, "a") as f:  #create the h5 file
            f.create_dataset('ppg', (0,DataLength), maxshape=(None,DataLength), chunks=(100, DataLength))
            f.create_dataset('label', (0,2), maxshape=(None,2), dtype=int, chunks=(100,2))
            f.create_dataset('subject_idx', (0,1), maxshape=(None,1), dtype=int, chunks=(100,1))
    FileDir = listdir(DataPath)
    #b,a = butter(5,[0.3, 4], 'bandpass', fs=25)
    testOUTPUT = np.empty((0,2))
    testPPG_RECORD = np.empty((0,DataLength))
    valOUTPUT = np.empty((0,2))
    valPPG_RECORD = np.empty((0,DataLength))
    trainOUTPUT = np.empty((0,2))
    trainPPG_RECORD = np.empty((0,DataLength))
    testSUB = np.empty((0,1))
    valSUB = np.empty((0,1))
    trainSUB = np.empty((0,1))
    for file in FileDir:
        DataInfo = splitext(file)[0].split('_')
        subjectID = int(DataInfo[0])
        SBP = int(DataInfo[1])
        DBP = int(DataInfo[2])
        df = pd.read_csv(join(DataPath,file))
        for i in (0,33):
            rppg = np.array(df.iloc[i*25:(i+7)*25,1])
            #rppg = filtfilt(b,a, rppg)
            rppg = rppg - np.mean(rppg)
            rppg = rppg / np.std(rppg)
            testOUTPUT = np.vstack((testOUTPUT,[SBP,DBP]))
            testPPG_RECORD = np.vstack((testPPG_RECORD,rppg))
            testSUB = np.vstack((testSUB, [subjectID]))
        for i in (3,30):
            rppg = np.array(df.iloc[i*25:(i+7)*25,1])
            #rppg = filtfilt(b,a, rppg)
            rppg = rppg - np.mean(rppg)
            rppg = rppg / np.std(rppg)
            valOUTPUT = np.vstack((valOUTPUT,[SBP,DBP]))
            valPPG_RECORD = np.vstack((valPPG_RECORD,rppg))
            valSUB = np.vstack((valSUB, [subjectID]))
        for i in range(8,28):
            rppg = np.array(df.iloc[i*25:(i+7)*25,1])
            #rppg = filtfilt(b,a, rppg)
            rppg = rppg - np.mean(rppg)
            rppg = rppg / np.std(rppg)
            trainOUTPUT = np.vstack((trainOUTPUT,[SBP,DBP]))
            trainPPG_RECORD = np.vstack((trainPPG_RECORD,rppg))
            trainSUB = np.vstack((trainSUB, [subjectID]))            
            # plt.clf()
            # plt.ion()
            # plt.plot(rppg, label="retrain")
            # plt.draw()
            # plt.pause(0.001)
            # key = keyboard.read_event(suppress=True).name
            # if key.lower() == 'q':
            #     plt.ioff()
            #     sys.exit(0)
    with h5py.File(TrainFile, "a") as f:
        BP_dataset = f['label']
        DatasetCurrLength = BP_dataset.shape[0]
        DatasetNewLength = DatasetCurrLength + trainOUTPUT.shape[0]
        BP_dataset.resize(DatasetNewLength, axis=0)
        BP_dataset[-trainOUTPUT.shape[0]:,:] = trainOUTPUT
        ppg_dataset = f['ppg']
        ppg_dataset.resize(DatasetNewLength, axis=0)
        ppg_dataset[-trainPPG_RECORD.shape[0]:,:] = trainPPG_RECORD
        subject_dataset = f['subject_idx']
        subject_dataset.resize(DatasetNewLength, axis=0)
        subject_dataset[-trainOUTPUT.shape[0]:,:] = trainSUB
        print(f'{trainOUTPUT.shape[0]} train samples ({DatasetNewLength} samples total)')
    with h5py.File(TestFile, "a") as f:
        BP_dataset = f['label']
        DatasetCurrLength = BP_dataset.shape[0]
        DatasetNewLength = DatasetCurrLength + testOUTPUT.shape[0]
        BP_dataset.resize(DatasetNewLength, axis=0)
        BP_dataset[-testOUTPUT.shape[0]:,:] = testOUTPUT
        ppg_dataset = f['ppg']
        ppg_dataset.resize(DatasetNewLength, axis=0)
        ppg_dataset[-testPPG_RECORD.shape[0]:,:] = testPPG_RECORD
        subject_dataset = f['subject_idx']
        subject_dataset.resize(DatasetNewLength, axis=0)
        subject_dataset[-testOUTPUT.shape[0]:,:] = testSUB
        print(f'{testOUTPUT.shape[0]} test samples ({DatasetNewLength} samples total)')   
    with h5py.File(ValFile, "a") as f:
        BP_dataset = f['label']
        DatasetCurrLength = BP_dataset.shape[0]
        DatasetNewLength = DatasetCurrLength + valOUTPUT.shape[0]
        BP_dataset.resize(DatasetNewLength, axis=0)
        BP_dataset[-valOUTPUT.shape[0]:,:] = valOUTPUT
        ppg_dataset = f['ppg']
        ppg_dataset.resize(DatasetNewLength, axis=0)
        ppg_dataset[-valPPG_RECORD.shape[0]:,:] = valPPG_RECORD
        subject_dataset = f['subject_idx']
        subject_dataset.resize(DatasetNewLength, axis=0)
        subject_dataset[-valOUTPUT.shape[0]:,:] = valSUB
        print(f'{valOUTPUT.shape[0]} val samples ({DatasetNewLength} samples total)')
        
if __name__ == "__main__":
    #prepare_retrain_dataset('.\\new_data', '.\\data\\retrain_data.h5', '.\\data\\retest_data.h5', '.\\data\\reval_data.h5', )
    chdir('D:\\RPPG')
    tfrecord_path = '.\\retrain'
    tfrecord_path_train = join(tfrecord_path, 'train')
    if not isdir(tfrecord_path_train):
        mkdir(tfrecord_path_train)
    tfrecord_path_val = join(tfrecord_path, 'val')
    if not isdir(tfrecord_path_val):
        mkdir(tfrecord_path_val)
    tfrecord_path_test = join(tfrecord_path, 'test')
    if not isdir(tfrecord_path_test):
        mkdir(tfrecord_path_test) 

    # train_idx = np.array(range(600))
    # test_idx = np.array(range(600,800))
    # val_idx = np.array(range(800,966))
    # rppg_train_hdf2tfrecord('.\\data\\retrain_data.h5',tfrecord_path_train,modus='train')
    rppg_hdf2tfrecord('.\\data\\retrain_data.h5',tfrecord_path_train,modus='train')
    rppg_hdf2tfrecord('.\\data\\retest_data.h5',tfrecord_path_test,modus='test')
    rppg_hdf2tfrecord('.\\data\\reval_data.h5',tfrecord_path_val,modus='val')
    # rppg_hdf2tfrecord('.\\data\\retrain_data.h5',tfrecord_path_val,idx=val_idx,modus='val')