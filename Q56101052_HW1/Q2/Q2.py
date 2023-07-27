import cv2
from os import listdir
from os.path import join
import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import biosppy

file_path = "EKG"
file_list = listdir(file_path)
dic = {}
dic["name"] = []
dic["heart_beats"] = []
for f in range(len(file_list)):
    img_list = sorted(listdir(join(file_path,file_list[f])))
    for d in range(len(img_list)):
        img_path = join(file_path,file_list[f],img_list[d])
        data_name = img_list[d].split('.')[0]
        print(data_name)
        img = cv2.imread(img_path)
        img = img[800:900,115:1350]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _ ,img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        peaks_array = np.zeros(img.shape[1])
        for k in range(img.shape[1]-1,0,-1):
            for l in range(img.shape[0]):
                if (img[l][k] == 0):
                    peaks_array[k] = img.shape[0] - l

        
        ecg = biosppy.signals.ecg.ecg(signal = peaks_array, sampling_rate = 100, show =False)
        rpeaks = ecg['rpeaks']
        print(len(rpeaks))
        heart_beats = len(rpeaks) * 6
        
        dic["name"].append('EKG_' + str(data_name)) 
        dic["heart_beats"].append(str(heart_beats)) 
        
        df = pd.DataFrame(dic)
        save_path = "heartbeat.csv"
        df.to_csv(save_path)