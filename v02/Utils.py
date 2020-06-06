import numpy as np
import librosa
import pandas as pd
from librosa import feature

n_mfcc = 100

def normalize(vector):
    return vector / np.linalg.norm(vector)

# def normalize_minmax_matrix(matrix):
#     for vector in matrix:
#         max_value = max(vector)
#         min_value = min(vector)
#         for i in range(len(vector)):
#             vector[i] = (vector[i] - min_value) / (max_value - min_value)
#     return matrix
# def normalize_minmax(vector):
#     max_value = max(vector)
#     min_value = min(vector)
#     for i in range(len(vector)):
#         vector[i] = (vector[i] - min_value) / (max_value - min_value)
#     return vector
def normalize_minmax_matrix(matrix):
    return [normalize_minmax(vector) for vector in matrix]

def normalize_minmax(vector):
    max_value = max(vector)
    min_value = min(vector)
    return [(vector[i] - min_value) / (max_value - min_value) for i in range(len(vector))]

def get_feature_vector(y, sr, file = None):
    # (1, frame)
    centroid = librosa.feature.spectral_centroid(y, sr)
    bandwidth = librosa.feature.spectral_bandwidth(y, sr)
    rolloff = librosa.feature.spectral_rolloff(y, sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y)
    # (number of mfcc feature, frame)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if file == None:
        return normalize_minmax_matrix(np.concatenate((centroid.T, bandwidth.T, rolloff.T,zcr.T, rms.T, mfccs.T), axis=1))

    # create array of name (1, frame)
    audio_name = np.full(bandwidth.shape, file)
    # (frame, 1 + 1 + 1 + 1 + number of mfcc feature)
    return np.concatenate((audio_name.T, centroid.T, bandwidth.T, rolloff.T, zcr.T, rms.T, mfccs.T), axis=1)

def loadAudio(file):
    y, sr = librosa.load(file, sr=48000)
    return y, sr

def toCsv(a, header, path):
    pd.DataFrame(a).to_csv(path, index=False, header=header)

y, sr = loadAudio("../audio/book.wav")
feat = get_feature_vector(y, sr, "haha")
# pd.DataFrame(feat).to_csv("test.csv", index=False)
nor = np.asarray(get_feature_vector(y, sr))

for n in nor:
    print(n)