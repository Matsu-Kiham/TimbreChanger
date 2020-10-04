# -*- coding: utf-8 -*-
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import scipy.signal as sig

import os
import sys
import numpy as np

from tkinter import *
from tkinter import ttk
import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog




root = tk.Tk()
root.title("TimbreChanger")
root.geometry("700x600")
fontstyle1 = tkFont.Font(family="Lucida Grande", size=28)
label1 = tk.Label(root, text="音色変更ツール", font=fontstyle1)
label1.place(x=220, y=50)





def sin_wave():
    typ = [('wavファイル','*.wav')] 
    dir = 'C:'
    fle = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

    fs, data = read(fle)

    if (data.shape[1] == 2):
        left = data[:, 0]
        right = data[:, 1]

    samplenum2 = data.shape[0]

    outdata_pre = np.arange(int(samplenum2 * 2), dtype= 'int16').reshape(samplenum2, 2)
    outdata = np.zeros_like(outdata_pre)
    outdata_w_pre = np.arange(int(samplenum2 * 2), dtype= 'int16').reshape(samplenum2, 2)
    outdata_w = np.zeros_like(outdata_w_pre)

    outdata_initialized_counter = 0

    i = 0
    j = 1
    frame_num = 4096
    fs_sep_num = 1
    fs_range = np.arange(12 * fs_sep_num)
    fs_range_pre = np.arange(12 * fs_sep_num, dtype=np.float64)
    
    if (startkeybox.get() == 'C'):
        startkey = 0
    if (startkeybox.get() == 'C#'):
        startkey = 1
    if (startkeybox.get() == 'D'):
        startkey = 2
    if (startkeybox.get() == 'D#'):
        startkey = 3
    if (startkeybox.get() == 'E'):
        startkey = 4
    if (startkeybox.get() == 'F'):
        startkey = 5
    if (startkeybox.get() == 'F#'):
        startkey = 6
    if (startkeybox.get() == 'G'):
        startkey = 7
    if (startkeybox.get() == 'G#'):
        startkey = 8
    if (startkeybox.get() == 'A'):
        startkey = 9
    if (startkeybox.get() == 'A#'):
        startkey = 10
    if (startkeybox.get() == 'B'):
        startkey = 11

    fs_range_pre[0] = 63 * ((1.059463094) ** startkey)
    fs_range_pre[1] = 65.406 * (1.059463094) ** (12 / fs_sep_num - 2 + 0.6)
    fs_range_pre[2] = 65.406 * (1.059463094) ** (12 / fs_sep_num + 0.7)
    fs_range_pre_counter = 3
    while fs_range_pre_counter < (12 * fs_sep_num - 2):
        fs_range_pre[fs_range_pre_counter] = fs_range_pre[fs_range_pre_counter - 2] * ((1.059463094) ** (12 / fs_sep_num))
        fs_range_pre[fs_range_pre_counter + 1] = fs_range_pre[fs_range_pre_counter - 1] * ((1.059463094) ** (12 / fs_sep_num))
        fs_range_pre_counter = fs_range_pre_counter + 2

    fs_range[12 * fs_sep_num - 1] = int(65.406 * ((1.059463094) ** (71 + startkey)) / fs * frame_num * 2)
    fs_range_counter = 0
    while fs_range_counter < (12 * fs_sep_num - 1):
        fs_range[fs_range_counter] = int(fs_range_pre[fs_range_counter] / fs * frame_num * 2)
        fs_range_counter = fs_range_counter + 1

    octave_check = np.arange(6, dtype=np.int8)

    if octave2_bln.get():
        octave_check[0] = 1
    else:
        octave_check[0] = 0
    
    if octave3_bln.get():
        octave_check[1] = 1
    else:
        octave_check[1] = 0

    if octave4_bln.get():
        octave_check[2] = 1
    else:
        octave_check[2] = 0

    if octave5_bln.get():
        octave_check[3] = 1
    else:
        octave_check[3] = 0

    if octave6_bln.get():
        octave_check[4] = 1
    else:
        octave_check[4] = 0

    if octave7_bln.get():
        octave_check[5] = 1
    else:
        octave_check[5] = 0


    
    frame1_sample = np.arange(int(frame_num * 2 - 1), dtype= 'int16')
    hw = 0.5 - 0.5 * np.cos(2 * 3.141593 * frame1_sample / (frame_num * 2))
    

    while i < (samplenum2 - frame_num * 2 - 1):

        
        frame1 = left[(2 * (j - 1) * frame_num):(-1 + 2 * j * frame_num)] * hw
        
        fourier1 = np.fft.fft(frame1)
        
        
        Amp1 = np.abs(fourier1/frame_num) 
        
        Amp1_sig_max_pre = np.arange(6 * fs_sep_num, dtype=np.float64)
        Amp1_sig_max_fs_pre = np.arange(6 * fs_sep_num, dtype=np.float64)
        Amp1_sig_max = np.zeros_like(Amp1_sig_max_pre)
        Amp1_sig_max_fs = np.zeros_like(Amp1_sig_max_fs_pre)
        Amp1_sig_counter = 0
        while Amp1_sig_counter < (6 * fs_sep_num):
            if (octave_check[int(Amp1_sig_counter / fs_sep_num)] == 1):
                Amp1_sig = Amp1[fs_range[Amp1_sig_counter * 2]:fs_range[Amp1_sig_counter * 2 + 1]]
                if (len(Amp1_sig) > 0):
                    Amp1_sig_max_index = np.argmax(Amp1_sig)
                    Amp1_sig_max[Amp1_sig_counter] = Amp1_sig[Amp1_sig_max_index] * 1
                    Amp1_sig_max_fs[Amp1_sig_counter] = (Amp1_sig_max_index + fs_range[Amp1_sig_counter * 2]) * fs / frame_num / 2
            Amp1_sig_counter += 1

        
        Amp1_sig_max_max_index = np.argmax(Amp1_sig_max)
        Amp1_sig_max_max = Amp1_sig_max[Amp1_sig_max_max_index]
        sw_counter = 0
        while sw_counter < (6 * fs_sep_num):
            if (Amp1_sig_max[sw_counter] > Amp1_sig_max_max * 0.001):
                sw_w_counter = 0
                while sw_w_counter < (frame_num * 2):
                    outdata[i + sw_w_counter, 0] = outdata[i + sw_w_counter, 0] + Amp1_sig_max[sw_counter] / 2 * np.sin(Amp1_sig_max_fs[sw_counter] * sw_w_counter / fs * 3.141593 * 2)
                    sw_w_counter += 1
            sw_counter += 1

        j += 1
        
        i = i + 2 * frame_num
   
    i = 0
    j = 1

    while i < (samplenum2 - frame_num * 2 - 1):
        frame2 = right[(2 * (j - 1) * frame_num):(-1 + 2 * j * frame_num)] * hw
        
        fourier2 = np.fft.fft(frame2)
        
        
        Amp2 = np.abs(fourier2/frame_num) 
        
        

        Amp2_sig_max_pre = np.arange(6 * fs_sep_num, dtype=np.float64)
        Amp2_sig_max_fs_pre = np.arange(6 * fs_sep_num, dtype=np.float64)
        Amp2_sig_max = np.zeros_like(Amp2_sig_max_pre)
        Amp2_sig_max_fs = np.zeros_like(Amp2_sig_max_fs_pre)
        Amp2_sig_counter = 0
        while Amp2_sig_counter < (6 * fs_sep_num):
            if (octave_check[int(Amp2_sig_counter / fs_sep_num)] == 1):
                Amp2_sig = Amp2[fs_range[Amp2_sig_counter * 2]:fs_range[Amp2_sig_counter * 2 + 1]]
                if (len(Amp2_sig) > 0):
                    Amp2_sig_max_index = np.argmax(Amp2_sig)
                    Amp2_sig_max[Amp2_sig_counter] = Amp2_sig[Amp2_sig_max_index] * 1
                    Amp2_sig_max_fs[Amp2_sig_counter] = (Amp2_sig_max_index + fs_range[Amp2_sig_counter * 2]) * fs / frame_num / 2
            Amp2_sig_counter += 1


        

        Amp2_sig_max_max_index = np.argmax(Amp2_sig_max)
        Amp2_sig_max_max = Amp2_sig_max[Amp2_sig_max_max_index]
        sw_counter = 0
        while sw_counter < (6 * fs_sep_num):
            if (Amp2_sig_max[sw_counter] > Amp2_sig_max_max * 0.001):
                sw_w_counter = 0
                while sw_w_counter < (frame_num * 2):
                    outdata[i + sw_w_counter, 1] = outdata[i + sw_w_counter, 1] + Amp2_sig_max[sw_counter] * np.sin(Amp2_sig_max_fs[sw_counter] * sw_w_counter / fs * 3.141593 * 2)
                    sw_w_counter += 1
            sw_counter += 1

        j += 1
        
        i = i + 2 * frame_num
    
    
    left_w = outdata[:, 0] / 1
    right_w = outdata[:, 1] / 1
    left_w_int16 = left_w.astype(np.int16)
    right_w_int16 = right_w.astype(np.int16)

    outdata_w[:,0] = left_w_int16
    outdata_w[:,1] = right_w_int16
    write('out.wav', fs, outdata_w)



def square_wave():
    typ = [('wavファイル','*.wav')] 
    dir = 'C:'
    fle = filedialog.askopenfilename(filetypes = typ, initialdir = dir)

    fs, data = read(fle)

    if (data.shape[1] == 2):
        left = data[:, 0]
        right = data[:, 1]

    samplenum2 = data.shape[0]

    outdata_pre = np.arange(int(samplenum2 * 2), dtype= 'int16').reshape(samplenum2, 2)
    outdata = np.zeros_like(outdata_pre)
    outdata_w_pre = np.arange(int(samplenum2 * 2), dtype= 'int16').reshape(samplenum2, 2)
    outdata_w = np.zeros_like(outdata_w_pre)

    outdata_initialized_counter = 0

    i = 0
    j = 1
    frame_num = 4096
    fs_sep_num = 1
    fs_range = np.arange(12 * fs_sep_num)
    fs_range_pre = np.arange(12 * fs_sep_num, dtype=np.float64)

    if (startkeybox.get() == 'C'):
        startkey = 0
    if (startkeybox.get() == 'C#'):
        startkey = 1
    if (startkeybox.get() == 'D'):
        startkey = 2
    if (startkeybox.get() == 'D#'):
        startkey = 3
    if (startkeybox.get() == 'E'):
        startkey = 4
    if (startkeybox.get() == 'F'):
        startkey = 5
    if (startkeybox.get() == 'F#'):
        startkey = 6
    if (startkeybox.get() == 'G'):
        startkey = 7
    if (startkeybox.get() == 'G#'):
        startkey = 8
    if (startkeybox.get() == 'A'):
        startkey = 9
    if (startkeybox.get() == 'A#'):
        startkey = 10
    if (startkeybox.get() == 'B'):
        startkey = 11


    fs_range_pre[0] = 63 * ((1.059463094) ** startkey)
    fs_range_pre[1] = 65.406 * (1.059463094) ** (12 / fs_sep_num - 2 + 0.6)
    fs_range_pre[2] = 65.406 * (1.059463094) ** (12 / fs_sep_num + 0.7)
    fs_range_pre_counter = 3
    while fs_range_pre_counter < (12 * fs_sep_num - 2):
        fs_range_pre[fs_range_pre_counter] = fs_range_pre[fs_range_pre_counter - 2] * ((1.059463094) ** (12 / fs_sep_num))
        fs_range_pre[fs_range_pre_counter + 1] = fs_range_pre[fs_range_pre_counter - 1] * ((1.059463094) ** (12 / fs_sep_num))
        fs_range_pre_counter = fs_range_pre_counter + 2

    fs_range[12 * fs_sep_num - 1] = int(65.406 * ((1.059463094) ** (71 + startkey)) / fs * frame_num * 2)
    fs_range_counter = 0
    while fs_range_counter < (12 * fs_sep_num - 1):
        fs_range[fs_range_counter] = int(fs_range_pre[fs_range_counter] / fs * frame_num * 2)
        fs_range_counter = fs_range_counter + 1

    octave_check = np.arange(6, dtype=np.int8)

    if octave2_bln.get():
        octave_check[0] = 1
    else:
        octave_check[0] = 0
    
    if octave3_bln.get():
        octave_check[1] = 1
    else:
        octave_check[1] = 0

    if octave4_bln.get():
        octave_check[2] = 1
    else:
        octave_check[2] = 0

    if octave5_bln.get():
        octave_check[3] = 1
    else:
        octave_check[3] = 0

    if octave6_bln.get():
        octave_check[4] = 1
    else:
        octave_check[4] = 0

    if octave7_bln.get():
        octave_check[5] = 1
    else:
        octave_check[5] = 0


    frame1_sample = np.arange(int(frame_num * 2 - 1), dtype= 'int16')
    hw = 0.5 - 0.5 * np.cos(2 * 3.141593 * frame1_sample / (frame_num * 2))
    

    while i < (samplenum2 - frame_num * 2 - 1):

        
        frame1 = left[(2 * (j - 1) * frame_num):(-1 + 2 * j * frame_num)] * hw
        
        fourier1 = np.fft.fft(frame1)
        
        
        Amp1 = np.abs(fourier1/frame_num) 
        
        
        Amp1_sig_max_pre = np.arange(6 * fs_sep_num, dtype=np.float64)
        Amp1_sig_max_fs_pre = np.arange(6 * fs_sep_num, dtype=np.float64)
        Amp1_sig_max = np.zeros_like(Amp1_sig_max_pre)
        Amp1_sig_max_fs = np.zeros_like(Amp1_sig_max_fs_pre)
        Amp1_sig_counter = 0
        while Amp1_sig_counter < (6 * fs_sep_num):
            if (octave_check[int(Amp1_sig_counter / fs_sep_num)] == 1):
                Amp1_sig = Amp1[fs_range[Amp1_sig_counter * 2]:fs_range[Amp1_sig_counter * 2 + 1]]
                if (len(Amp1_sig) > 0):
                    Amp1_sig_max_index = np.argmax(Amp1_sig)
                    Amp1_sig_max[Amp1_sig_counter] = Amp1_sig[Amp1_sig_max_index] * 1
                    Amp1_sig_max_fs[Amp1_sig_counter] = (Amp1_sig_max_index + fs_range[Amp1_sig_counter * 2]) * fs / frame_num / 2
            Amp1_sig_counter += 1

        

        Amp1_sig_max_max_index = np.argmax(Amp1_sig_max)
        Amp1_sig_max_max = Amp1_sig_max[Amp1_sig_max_max_index]
        sw_counter = 0
        while sw_counter < (6 * fs_sep_num):
            if (Amp1_sig_max[sw_counter] > Amp1_sig_max_max * 0.001):
                sw_w_counter = 0
                posinegacheck = 0
                while sw_w_counter < (frame_num * 2):
                    if (posinegacheck < (fs / Amp1_sig_max_fs[sw_counter] / 2)):
                        outdata[i + sw_w_counter, 0] = outdata[i + sw_w_counter, 0] + Amp1_sig_max[sw_counter]
                    if (posinegacheck > (fs / Amp1_sig_max_fs[sw_counter] / 2)):
                        outdata[i + sw_w_counter, 0] = outdata[i + sw_w_counter, 0] - Amp1_sig_max[sw_counter]
                    if (posinegacheck > (fs / Amp1_sig_max_fs[sw_counter])):
                        posinegacheck = 0
                    
                    sw_w_counter += 1
                    posinegacheck += 1
            sw_counter += 1

        j += 1
        
        i = i + 2 * frame_num
   
    i = 0
    j = 1

    while i < (samplenum2 - frame_num * 2 - 1):
        frame2 = right[(2 * (j - 1) * frame_num):(-1 + 2 * j * frame_num)] * hw
        
        fourier2 = np.fft.fft(frame2)
        
        
        Amp2 = np.abs(fourier2/frame_num) 
        
        
        Amp2_sig_max_pre = np.arange(6 * fs_sep_num, dtype=np.float64)
        Amp2_sig_max_fs_pre = np.arange(6 * fs_sep_num, dtype=np.float64)
        Amp2_sig_max = np.zeros_like(Amp2_sig_max_pre)
        Amp2_sig_max_fs = np.zeros_like(Amp2_sig_max_fs_pre)
        Amp2_sig_counter = 0
        while Amp2_sig_counter < (6 * fs_sep_num):
            if (octave_check[int(Amp2_sig_counter / fs_sep_num)] == 1):
                Amp2_sig = Amp2[fs_range[Amp2_sig_counter * 2]:fs_range[Amp2_sig_counter * 2 + 1]]
                if (len(Amp2_sig) > 0):
                    Amp2_sig_max_index = np.argmax(Amp2_sig)
                    Amp2_sig_max[Amp2_sig_counter] = Amp2_sig[Amp2_sig_max_index] * 1
                    Amp2_sig_max_fs[Amp2_sig_counter] = (Amp2_sig_max_index + fs_range[Amp2_sig_counter * 2]) * fs / frame_num / 2
            Amp2_sig_counter += 1


        

        Amp2_sig_max_max_index = np.argmax(Amp2_sig_max)
        Amp2_sig_max_max = Amp2_sig_max[Amp2_sig_max_max_index]
        sw_counter = 0
        while sw_counter < (6 * fs_sep_num):
            if (Amp2_sig_max[sw_counter] > Amp2_sig_max_max * 0.001):
                sw_w_counter = 0
                posinegacheck = 0
                while sw_w_counter < (frame_num * 2):
                    if (posinegacheck < (fs / Amp2_sig_max_fs[sw_counter] / 2)):
                        outdata[i + sw_w_counter, 1] = outdata[i + sw_w_counter, 0] + Amp2_sig_max[sw_counter]
                    if (posinegacheck > (fs / Amp2_sig_max_fs[sw_counter] / 2)):
                        outdata[i + sw_w_counter, 1] = outdata[i + sw_w_counter, 0] - Amp2_sig_max[sw_counter]
                    if (posinegacheck > (fs / Amp2_sig_max_fs[sw_counter])):
                        posinegacheck = 0
                    sw_w_counter += 1
                    posinegacheck += 1
            sw_counter += 1

        j += 1
        i = i + 2 * frame_num
    
    left_w = outdata[:, 0] / 1
    right_w = outdata[:, 1] / 1
    left_w_int16 = left_w.astype(np.int16)
    right_w_int16 = right_w.astype(np.int16)

    outdata_w[:,0] = left_w_int16
    outdata_w[:,1] = right_w_int16
    write('out.wav', fs, outdata_w)

    
    
fontstyle2 = tkFont.Font(family="Lucida Grande", size=15)
button1 = tk.Button(root, text="正弦波", bg='#ff7f50', width=20, font=fontstyle2, command=sin_wave)
button1.place(x=230, y=300)
button2 = tk.Button(root, text="矩形波", bg='#f0e68c', width=20, font=fontstyle2, command=square_wave)
button2.place(x=230, y=350)
fontstyle3 = tkFont.Font(family="Lucida Grande", size=16)
label2 = tk.Label(root, text="音を拾う範囲", font=fontstyle3)
label2.place(x=10, y=140)
label3 = tk.Label(root, text="（オクターブ）", font=fontstyle3)
label3.place(x=10, y=170)
fontstyle4 = tkFont.Font(family="Lucida Grande", size=14)
octave2_bln = tk.BooleanVar()
octave2_check = tk.Checkbutton(root, text='2', font=fontstyle4, variable=octave2_bln)
octave2_check.place(x=10, y=200)
octave3_bln = tk.BooleanVar()
octave3_check = tk.Checkbutton(root, text='3', font=fontstyle4, variable=octave3_bln)
octave3_check.place(x=10, y=230)
octave4_bln = tk.BooleanVar()
octave4_check = tk.Checkbutton(root, text='4', font=fontstyle4, variable=octave4_bln)
octave4_check.place(x=10, y=260)
octave5_bln = tk.BooleanVar()
octave5_check = tk.Checkbutton(root, text='5', font=fontstyle4, variable=octave5_bln)
octave5_check.place(x=10, y=290)
octave6_bln = tk.BooleanVar()
octave6_check = tk.Checkbutton(root, text='6', font=fontstyle4, variable=octave6_bln)
octave6_check.place(x=10, y=320)
octave7_bln = tk.BooleanVar()
octave7_check = tk.Checkbutton(root, text='7', font=fontstyle4, variable=octave7_bln)
octave7_check.place(x=10, y=350)
label4 = tk.Label(root, text="開始音", font=fontstyle3)
label4.place(x=10, y=380)
startkeylist = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
startkeyname = StringVar()
startkeybox = ttk.Combobox(root, state='readonly', values=startkeylist, width=10, font=fontstyle4, textvariable=startkeyname)
startkeybox.set(startkeylist[0])
startkeybox.place(x=10, y=410)

root.mainloop()