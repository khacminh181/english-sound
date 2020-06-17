import numpy as np

def ZeroCrossingRate(audio_frames):
    """
    Đầu vào là array các frame của file audio
    :param audio_frames: Input array
    :return: np.ndarray [shape=(1, t)]
       Với t là số frame
    """

    frame_length = audio_frames.shape[0]
    audio_frames = audio_frames.T
    # Tạo 1 boolean array tương ứng khi signal > 0  (np.sign(frame) > 0)
    # So sánh sự khác biệt sử dụng np.diff để lấy số lần đổi dấu
    # Chia cho frame_length để lấy rate
    return np.array([(np.sum(np.diff(np.sign(frame) > 0))) / frame_length for frame in audio_frames])

