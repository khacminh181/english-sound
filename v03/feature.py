import numpy as np

def zero_crossing_rate(y, frame_length=2048, hop_length=512, axis=-1):
    """
    Tính tần suất cắt trục hoành
    Đầu vào là array các frame của file audio    :param audio_frames: np.ndarray [shape=(frame_length, t)] Input array
    :return: np.ndarray [shape=(1, t)]
       Với t là số frame
    """
    ypad = np.pad(y, int(frame_length // 2), mode='edge')

    audio_frames = frame(ypad, frame_length, hop_length)
    frame_length = audio_frames.shape[0]
    audio_frames = audio_frames.T
    # Tạo 1 boolean array tương ứng khi signal > 0  (np.sign(frame) > 0)
    # So sánh sự khác biệt sử dụng np.diff để lấy số lần đổi dấu
    # Chia cho frame_length để lấy rate
    return np.array([(np.sum(np.diff(np.sign(f) > 0))) / frame_length for f in audio_frames]).reshape(1, -1)

def energy(y, frame_length=2048, hop_length=512, axis=-1):
    """
    Tính năng lượng trung bình
    :param audio_frames: np.ndarray [shape=(frame_length, t)] Input array
    :return: np.ndarray [shape=(1, t)]
       Với t là số frame
    """

    ypad = np.pad(y, int(frame_length // 2), mode='edge')
    # (2048, t)
    audio_frames = frame(ypad, frame_length, hop_length)
    # E = x^2 / N
    return np.mean(np.abs(audio_frames)**2, axis=0, keepdims=True)

def frame(x, frame_length=2048, hop_length=512, axis=-1):
    """
    Slide data array thành overlapping array frames.
    Ex: x = [0, 1, 2, 3, 4, 5, 6]
    axis = -1
    [[0, 2, 4],
     [1, 3, 5],
     [2, 4, 6]]
     axis = 0
     [[0, 1, 2],
     [2, 3, 4],
     [4, 5, 6]]
    :param x:
    :param frame_length:
    :param hop_length:
    :param axis:
    :return:
    """
    # Tính số frame đầu ra
    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    # số bytes để nhảy sang điểm mới
    # Giả dụ có matrix (3,3), với axis=1, đi từ điểm (0,0)->(0,1) mất 4 bytes, Với axis=0, đi từ điểm (0,0)->(1,0) sẽ phải đi (0,0)->(0,1)->(0,2)->(1,0), vì vậy mất 4*3 = 12 bytes.
    # Vì vậy vơi matrix (3,3) strides là (12, 4)
    # strides (shape[1] * itemsize, itemsize)
    strides = np.asarray(x.strides)
    # itemsize - length của 1 phần tử trong array
    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize
    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]
    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
