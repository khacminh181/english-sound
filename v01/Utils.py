import numpy as np
import librosa
from librosa import feature

n_mfcc = 256

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

fn_list_i = [
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff
]

fn_list_ii = [
    feature.zero_crossing_rate,
    feature.rms
]

def get_feature_vector(y, sr, file):
    feat_vect_i = [np.mean(funct(y, sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfccs)
    new = np.vstack([mfccs, mfcc_delta])
    mfccs_processed = np.mean(new.T, axis=0).tolist()
    feature_vector = [file] + feat_vect_i + feat_vect_ii + mfccs_processed
    # for e in mfcc:
    #     feature_vector += [np.mean(e)]
    return feature_vector


def audio_2_feature_vector(y, sr):
    feat_vect_i = [np.mean(funct(y, sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfccs)
    new = np.vstack([mfccs, mfcc_delta])
    mfccs_processed = np.mean(new.T, axis=0).tolist()
    feature_vector = feat_vect_i + feat_vect_ii + mfccs_processed
    # feature_vector = feat_vect_i + feat_vect_ii
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # for e in mfcc:
    #     feature_vector += [np.mean(e)]
    return normalize_minmax(feature_vector)

def loadAudio(file):
    y, sr = librosa.load(file, duration=30, sr=16000)
    return y, sr
