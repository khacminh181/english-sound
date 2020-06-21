import pandas as pd
from v03.feature import *
from glob import glob
import librosa

def normalize_minmax_matrix(matrix):
    return [normalize_minmax(vector) for vector in matrix]

def normalize_minmax(vector):
    max_value = np.max(vector)
    min_value = np.min(vector)
    # print('max: {}, min: {}'.format(max_value, min_value))
    return np.array([(2 * (vector[i] - min_value) / (max_value - min_value)) - 1 for i in range(len(vector))])

def get_feature_vector(y, sr, file = None):
    # (1, frame)
    centroid = librosa.feature.spectral_centroid(y, sr)
    bandwidth = librosa.feature.spectral_bandwidth(y, sr)
    rolloff = librosa.feature.spectral_rolloff(y, sr)
    zcr = zero_crossing_rate(y)
    rms = energy(y)

    # Get n_frame
    n_frame = zcr.shape[1]

    centroid = normalize_minmax(centroid)
    bandwidth = normalize_minmax(bandwidth)
    rolloff = normalize_minmax(rolloff)
    zcr = normalize_minmax(zcr)
    rms = normalize_minmax(rms)
    norm_vector = np.concatenate((centroid, bandwidth, rolloff, zcr, rms), axis=-1)
    # norm_vector = normalize_minmax(norm_vector)
    if file == None:
        return norm_vector
    return n_frame, np.append([file], norm_vector)

def loadAudio(file):
    y_max = 20480
    duration_max = 2.56
    y, sr = librosa.load(file, sr=8000, duration=duration_max)
    if y.shape[0] < y_max:
        y = np.pad(y, (0, y_max - len(y) % y_max), 'constant')

    return y, sr

def toCsv(a, header, path):
    pd.DataFrame(a).to_csv(path, index=False, header=header)

def getAudioName(x):
    return x.split('/')[-1].split('_')[0]

# Get list audio feature of all word sound file
def findMax(audio_files):
    ys = []
    durations = []
    for file in audio_files:
        y, sr = librosa.load(file, sr=8000)
        duration = librosa.get_duration(y, sr)
        durations.append(duration)
        ys.append(y.shape[0])
    return max(ys), max(durations)

def initalize():
    # directories of normal audios
    data_dir = "../train/"
    audio_files = glob(data_dir + '*.wav')
    print(audio_files)

    # find max sample
    y_max, duration_max = findMax(audio_files)
    return audio_files, y_max, duration_max

y, sr = loadAudio("../audio/book.wav")
t = get_feature_vector(y, sr)
duration = librosa.get_duration(y, sr)

