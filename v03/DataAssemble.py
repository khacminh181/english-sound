from v02.Utils import *
from v03.feature import *
from IPython.display import Audio
y, sr = loadAudio("../audio/book.wav")
Audio(y, rate=sr)

duration = librosa.get_duration(y, sr)
c = librosa.feature.spectral_centroid(y, sr)
spectrogram = librosa.core.spectrum._spectrogram(y=y, )
stft2 = librosa.core.spectrum.stft(y)

zcr = librosa.feature.zero_crossing_rate(y)


ypad = np.pad(y, int(2048 // 2), mode='edge')

frame1 = librosa.util.frame(ypad, 2048, 512)
frame2 = frame(ypad, 2048, 512)
zcrr = librosa.core.audio.zero_crossings(frame1, axis=0, pad=None)
my_zcr = np.diff(np.sign(frame1.T) > 0).T
my_zcr1 = zero_crossing_rate(y)
e = energy(y)
rmse = librosa.feature.rms(y)
z = np.mean(zcrr, axis=0, keepdims=True)
true = np.sum(zcrr[:,0])
true2 = np.sum(my_zcr[:,0])
spectrum = librosa.stft(y)

# test = normalize_minmax(zcr)
max = np.max(zcr)

freq = librosa.core.fft_frequencies(sr=sr, n_fft=2048)