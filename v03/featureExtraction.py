from v03.Utils import *

def header(n_frame):
    header = 'filename'
    for i in range(1, n_frame + 1):
        header += f' spectral_centroid{i}'
    for i in range(1, n_frame + 1):
        header += f' spectral_bandwidth{i}'
    for i in range(1, n_frame + 1):
        header += f' rolloff{i}'
    for i in range(1, n_frame + 1):
        header += f' zero_crossing_rate{i}'
    for i in range(1, n_frame + 1):
        header += f' energy{i}'
    return header.split()

# Get list audio feature of all word sound file
def getAllFeatures(audio_files):
    audios_feat = []
    n_frame = 0
    for file in audio_files:
        y, sr = loadAudio(file)
        n_frame,feature = get_feature_vector(y, sr, file)
        audios_feat.append(feature)
    return n_frame, np.vstack(audios_feat)

audio_files, y_max, duration_max = initalize()

# Save all the features to the csv file
output = "features8.csv"
n_frame, features = getAllFeatures(audio_files)
toCsv(features, header=header(n_frame), path=output)

print("DONE")