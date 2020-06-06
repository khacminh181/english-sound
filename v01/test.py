import os
from tqdm import tqdm
import numpy as np
from python_speech_features import mfcc, fbank, logfbank
import librosa
import librosa
from librosa import feature
from glob import glob



y , sr = librosa.load("../audio/book.wav",duration=1, sr=48000)
print(y.shape)
y2 , sr2 = librosa.load("../audio/agreement--_gb_1.wav", duration=1, sr=16000)
data = librosa.feature.mfcc(y, sr, n_mfcc=40)
data2 = librosa.feature.mfcc(y2, sr, n_mfcc=100)

print("Sample rate: {0}Hz".format(sr))
print("Audio duration: {0}s".format(len(y) / sr))
#
# mfccs_processed = np.mean(data.T, axis=0)
# print(mfccs_processed.tolist())

# mfcc_delta = librosa.feature.delta(data)
# new = np.vstack([data, mfcc_delta])

def crop_feature(feat, i = 0, nb_step=10, maxlen=100):
    print(np.array(feat[i : i + nb_step]).shape)
    crop_feat = np.array(feat[i : i + nb_step]).flatten()
    print(crop_feat.shape)
    print(crop_feat)
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    print(crop_feat)
    return crop_feat

features = []
feat = mfcc(y, sr, nfilt=10, winstep=0.02)
for i in range(0, feat.shape[0] - 10, 5):
    print(i)
    x = crop_feature(feat, i, nb_step=10)
    print(x.shape)
    features.append(x)
print("shape {}".format(librosa.feature.rms(y).shape))
centroid = feature.spectral_centroid(y, sr)
print(centroid.shape)
bandwidth = feature.spectral_bandwidth(y, sr)
print(bandwidth.shape)
print(feature.spectral_rolloff(y, sr).shape)
print(feature.rms(y).shape)
name = np.full(bandwidth.shape, "haha")

max = np.concatenate((name.T, centroid.T, bandwidth.T, data.T), axis=1)

#
# test = np.zeros((41, 10))
# print(test.shape)
# test =test[0 : 11]
# print(test.shape)