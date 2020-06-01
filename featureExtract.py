import os
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
import librosa
from librosa import feature
from glob import glob


#directories of normal audios
data_dir = "audio/"
audio_files = glob(data_dir + '*.wav')

print(audio_files)

fn_list_i = [
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
]

fn_list_ii = [
    feature.zero_crossing_rate
]


def get_feature_vector1(y, sr, file):
    feat_vect_i = [np.mean(funct(y, sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]
    feature_vector = [file] + feat_vect_i + feat_vect_ii
    return feature_vector

def get_feature_vector(y, sr):
    feat_vect_i = [np.mean(funct(y, sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]
    feature_vector = feat_vect_i + feat_vect_ii
    return feature_vector


audios_feat = []
for file in audio_files:
   y , sr = librosa.load(file, sr=48000)
   feature_vector = get_feature_vector1(y, sr, file)
   # feature_vector = get_feature_vector(y, sr)
   audios_feat.append(feature_vector)

import csv
output = "features_00.csv"
header =["file", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"]
with open(output,"+w") as f:
    csv_writer = csv.writer(f, delimiter=",")
    csv_writer.writerow(header)
    csv_writer.writerows(audios_feat)