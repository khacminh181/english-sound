from glob import glob
from v01.Utils import *

header = 'filename spectral_centroid spectral_bandwidth rolloff zero_crossing_rate rmse'
for i in range(1, n_mfcc + 1):
    header += f' mfcc{i}'
header = header.split()

#directories of normal audios
data_dir = "audio/"
audio_files = glob(data_dir + '*.wav')
print(audio_files)

audios_feat = []
for file in audio_files:
   y , sr = loadAudio(file)
   feature_vector = get_feature_vector(y, sr, file)
   # feature_vector = get_feature_vector(y, sr)
   audios_feat.append(feature_vector)

import csv
output = "featuresplayground100-1.csv"
# header =["file", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"]
with open(output,"+w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)
    csv_writer.writerows(audios_feat)