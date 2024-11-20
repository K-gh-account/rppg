# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 13:20:59 2024

@author: kenne
"""

import cv2
import time
from threading import Timer
from queue import Queue
num_frames = 0

def read_frame():
    global num_frames
    ret,frame = video.read()
    frame_queue.put_nowait(frame)
    if num_frames < 119:
        t = Timer(0.02,read_frame)
        t.start()
    num_frames += 1

        
if __name__ == '__main__' :
 
    # Start default camera
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW);
    video.set(cv2.CAP_PROP_FPS, 25)
    frame_queue = Queue(maxsize = 200)
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

 
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
 
    if int(major_ver)  < 3 :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        
    start = time.time()

    while True:
        if num_frames >=175:
            print('stop')
            break
        ret,frame = video.read()
        frame_queue.put(frame)
        num_frames += 1
        
    print("Capturing {0} frames".format(num_frames))

    end = time.time()
 
    # Time elapsed
    seconds = end - start
    print ("Time taken : {0} seconds".format(seconds))

    # Calculate frames per second
    fps  = num_frames / seconds
    print("Estimated frames per second : {0}".format(fps))
    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    videoWrite = cv2.VideoWriter(r'test.mp4', fourcc, fps, (width,height))
    
    for i in range(175):
        save_frame = frame_queue.get()
        videoWrite.write(save_frame)
    
    # Release video
    video.release()
    videoWrite.release()