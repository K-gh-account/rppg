# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:42:42 2024

@author: kenne
"""

import matplotlib.pyplot as plt
import numpy as np
import keyboard
import time

def plot_sin(amplitude):
    plt.clf()  # 清除当前图形
    x = np.linspace(0, 2*np.pi, 1000)
    y = amplitude * np.sin(x)
    
    plt.plot(x, y)
    plt.title(f'Sin Function (Amplitude: {amplitude:.2f})')
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.xlim(0, 2*np.pi)
    plt.ylim(-max(amplitude*1.2, 1), max(amplitude*1.2, 1))
    
    plt.draw()
    plt.pause(0.001)  # 需要暂停一小段时间来显示图形

def main():
    amplitude = 1.0
    plt.ion()  # 打开交互模式
    plt.figure()
    
    while True:
        plot_sin(amplitude)
        print("按 'Q' 退出，按其他键增加振幅")
        
        key = keyboard.read_event(suppress=True).name
        if key.lower() == 'q':
            break
        
        amplitude *= 1.5
    
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()