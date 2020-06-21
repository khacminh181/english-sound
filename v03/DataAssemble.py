from v03.Utils import *
from v03.feature import *
from IPython.display import Audio
# y, sr = loadAudio("../audio/book.wav")
# Audio(y, rate=sr)
#
# duration = librosa.get_duration(y, sr)
# c = librosa.feature.spectral_centroid(y, sr)
# spectrogram = librosa.core.spectrum._spectrogram(y=y, )
# stft2 = librosa.core.spectrum.stft(y)
#
# zcr = librosa.feature.zero_crossing_rate(y)
#
#
# ypad = np.pad(y, int(2048 // 2), mode='edge')
#
# frame1 = librosa.util.frame(ypad, 2048, 512)
# frame2 = frame(ypad, 2048, 512)
# zcrr = librosa.core.audio.zero_crossings(frame1, axis=0, pad=None)
# my_zcr = np.diff(np.sign(frame1.T) > 0).T
# my_zcr1 = zero_crossing_rate(y)
# e = energy(y)
# rmse = librosa.feature.rms(y)
# z = np.mean(zcrr, axis=0, keepdims=True)
# true = np.sum(zcrr[:,0])
# true2 = np.sum(my_zcr[:,0])
# spectrum = librosa.stft(y)
#
# # test = normalize_minmax(zcr)
# max = np.max(zcr)
#
# freq = librosa.core.fft_frequencies(sr=sr, n_fft=2048)

y, sr = loadAudio("../audio/book.wav")
ypad = np.pad(y, int(2048 // 2), mode='edge')
frame_length=2048
hop_length=512
n_frames = 1 + (ypad.shape[-1] - frame_length) // hop_length
# số bytes để nhảy sang điểm mới
# Giả dụ có matrix (3,3), với axis=1, đi từ điểm (0,0)->(0,1) mất 4 bytes, Với axis=0, đi từ điểm (0,0)->(1,0) sẽ phải đi (0,0)->(0,1)->(0,2)->(1,0), vì vậy mất 4*3 = 12 bytes.
# Vì vậy vơi matrix (3,3) strides là (12, 4)
# strides (shape[1] * itemsize, itemsize)
strides1 = np.asarray(ypad.strides)
# itemsize - length của 1 phần tử trong array
new_stride = np.prod(strides1[strides1 > 0] // ypad.itemsize) * ypad.itemsize
shape = list(ypad.shape)[:-1] + [frame_length, n_frames]
print(list(ypad.shape)[:-1])
strides = list(strides1) + [hop_length * new_stride]
print([hop_length * new_stride])
frame2 = frame(ypad, 2048, 512)