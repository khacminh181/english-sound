import numpy as np
import librosa
import pandas as pd
from librosa import feature, display
import matplotlib.pyplot as plt

n_mfcc = 13

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

def get_feature_vector(y, sr, file = None, MFCC = False):
    if MFCC == True:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if file == None:
            return normalize_minmax_matrix(mfccs.T)

        # create array of name (1, frame)
        audio_name = np.full((1 ,mfccs.shape[1]), getAudioName(file))
        # (frame, 1 + 1 + 1 + 1 + number of mfcc feature)
        return np.concatenate((audio_name.T, mfccs.T), axis=1)

    else:
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
    y, sr = librosa.load(file, sr=8000)
    # yt, index = librosa.effects.trim(y)
    return y, sr

def toCsv(a, header, path):
    pd.DataFrame(a).to_csv(path, index=False, header=header)

def getAudioName(x):
    return x.split('/')[-1].split('_')[0]

# y, sr = loadAudio("../audio/book.wav")
# duration = librosa.get_duration(y, sr)
# c = librosa.feature.spectral_centroid(y, sr)
# spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
# zcr = librosa.feature.zero_crossing_rate(y)
# mfcc = librosa.feature.mfcc(y, sr)
# ypad = np.pad(y, int(2048 // 2), mode='edge')
# frame = librosa.util.frame(ypad, 2048, 512)
#
# spectrum = librosa.stft(y)


# plt.figure()
# plt.subplot(3, 1, 1)
# x = librosa.display.waveplot(y, sr=sr)

# plt.figure(figsize=(12, 8))
# D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
# plt.subplot(4, 2, 1)
# librosa.display.specshow(D, y_axis='linear')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Linear-frequency power spectrogram')
#
#
# log_S = librosa.amplitude_to_db(spectrogram, ref=np.max)
#
# # Make a new figure
# plt.figure(figsize=(12,4))
#
# # Display the spectrogram on a mel scale
# # sample rate and hop length parameters are used to render the time axis
# librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
#
# # Put a descriptive title on the plot
# plt.title('mel power spectrogram')
#
# # draw a color bar
# plt.colorbar(format='%+02.0f dB')
#
# # Make the figure layout compact
# plt.tight_layout()
#
# plt.show()


# feat = get_feature_vector(y, sr, MFCC=True)
# # pd.DataFrame(feat).to_csv("test.csv", index=False)
# nor = np.asarray(get_feature_vector(y, sr))
#
# for n in nor:
#     print(n)
