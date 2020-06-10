from glob import glob
from v02.Utils import *

def header(n_mfcc):
    header = 'filename spectral_centroid spectral_bandwidth rolloff zero_crossing_rate rmse'
    # header = 'filename'
    for i in range(1, n_mfcc + 1):
        header += f' mfcc{i}'
    return header.split()

# Get list audio feature of all word sound file
def getAllFeatures(audio_files):
    audios_feat = []
    for file in audio_files:
        y, sr = loadAudio(file)
        feature = get_feature_vector(y, sr, file)
        audios_feat.append(feature)
    return np.vstack(audios_feat)

#directories of normal audios
data_dir = "../train/"
audio_files = glob(data_dir + '*.wav')
print(audio_files)

# Save all the features to the csv file
output = "features-v02-02.csv"
features = getAllFeatures(audio_files)
toCsv(features, header=header(n_mfcc), path=output)

print("DONE")