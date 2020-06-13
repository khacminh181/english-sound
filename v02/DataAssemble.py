from v02.Utils import *

y, sr = loadAudio("../audio/book.wav")
duration = librosa.get_duration(y, sr)
c = librosa.feature.spectral_centroid(y, sr)
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y, sr)
ypad = np.pad(y, int(2048 // 2), mode='edge')
frame = librosa.util.frame(ypad, 2048, 512)

spectrum = librosa.stft(y)


