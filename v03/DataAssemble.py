from v02.Utils import *
from v03.feature import *
from collections import Counter

y, sr = loadAudio("../audio/book.wav")
duration = librosa.get_duration(y, sr)
c = librosa.feature.spectral_centroid(y, sr)
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y, sr)


ypad = np.pad(y, int(2048 // 2), mode='edge')
frame = librosa.util.frame(ypad, 2048, 512)
zcrr = librosa.core.audio.zero_crossings(frame, axis=0, pad=None)
my_zcr = np.diff(np.sign(frame.T) > 0).T
my_zcr1 = ZeroCrossingRate(frame)
z = np.mean(zcrr, axis=0, keepdims=True)
true = np.sum(zcrr[:,0])
true2 = np.sum(my_zcr[:,0])
spectrum = librosa.stft(y)

# test = normalize_minmax(zcr)
max = np.max(zcr)

freq = librosa.core.fft_frequencies(sr=sr, n_fft=2048)