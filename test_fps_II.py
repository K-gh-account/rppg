# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:29:18 2024

@author: kenne
"""

import cv2
import queue
import threading
import time
import numpy as np
from numba import jit

frame_queue = queue.Queue(maxsize=30)
display_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

frames_processed = 0
start_time = time.time()

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

@jit(nopython=True)
def camera_thread():
    global cap
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if frame_queue.qsize() > 10:
                continue
            frame_queue.put(frame)
        else:
            break
    print("Camera thread finished")

@jit(nopython=True)
def processing_thread():
    global frames_processed, start_time
    start_time = time.time()
    last_print_time = start_time

    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            frames_processed += 1
            
            if frames_processed % 5 == 0:
                if display_queue.full():
                    display_queue.get()  # Remove old frame
                display_queue.put(processed_frame)
            
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                fps = frames_processed / (current_time - start_time)
                print(f"Current FPS: {fps:.2f}")
                last_print_time = current_time

    print("Processing thread finished")

@jit(nopython=True)
def display_thread():
    cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
    while not stop_event.is_set():
        if not display_queue.empty():
            frame = display_queue.get()
            cv2.imshow('Processed Frame', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
    cv2.destroyAllWindows()
    print("Display thread finished")


def main():
    global cap, frames_processed, start_time

    camera_thread_obj = threading.Thread(target=camera_thread)
    processing_thread_obj = threading.Thread(target=processing_thread)
    display_thread_obj = threading.Thread(target=display_thread)

    camera_thread_obj.start()
    processing_thread_obj.start()
    display_thread_obj.start()

    # Wait for threads to finish
    camera_thread_obj.join()
    processing_thread_obj.join()
    display_thread_obj.join()

    # Cleanup
    cap.release()

    # Calculate final statistics
    total_time = time.time() - start_time
    actual_fps = frames_processed / total_time

    print(f"Total frames processed: {frames_processed}")
    print(f"Total running time: {total_time:.2f} seconds")
    print(f"Actual FPS: {actual_fps:.2f}")

if __name__ == "__main__":
    main()
    print("Program finished")