from annoy import AnnoyIndex
from v02.Utils import *
from collections import Counter
import pandas as pd

df = pd.read_csv('./feature/features-final.csv')
# List file name
words = df["filename"].tolist()

def get_audio_feature_vector(audio):
    y, sr = loadAudio(audio)
    feat = get_feature_vector(y, sr)
    return np.array(feat)

def get_result(feat, words, searchK):
    results = []
    for vector_feat in feat:
        result = u.get_nns_by_vector(vector_feat, n=searchK)
        result = [getAudioName(words[k]) for k in result]
        results.append(result)
    return np.array(results).flatten()

def Result(audio, results):
    print("Search for word {}".format(audio))
    print("RESULT")
    most_song = Counter(results)
    return most_song.most_common()

def printResult(audio, results):
    r = Result(audio, results)
    # print(r)
    print('Result word is: {}'.format(r[0][0]))
    print('Result word 2 is: {}'.format(r[1][0]))

# load index tree
f = len(df.columns) - 1
u = AnnoyIndex(f, metric='euclidean')
u.load('./indextree/features-minmax-final.ann')

# search
# audio = "../audio/minimal--_gb_1.wav"
# audio = "../audio/book.wav"
# audio = "../audio-test/book--_us_1 (online-audio-converter.com).wav"
# audio = "../audio-test/ordinary--_us_1 (online-audio-converter.com).wav"
# audio = "../train/agreement_en(2).wav"

# audio = "../test/agreement_en(1).wav"
# audio = "../test/book_en(3).wav"
# audio = "../test/circle_en(1).wav"
# audio = "../test/international_en.wav"
# audio = "../test/minimal_en(1).wav"
# audio = "../test/ordinary_en(1).wav"
audio = "../test/pronunciation_en_swing (3).wav"
# audio = "../audio-test/ordinary--_us_1 (online-audio-converter.com).wav"
# audio = "../audio-test/book--_us_1 (online-audio-converter.com).wav"

feat = get_audio_feature_vector(audio)
results = get_result(feat, words, 5)
printResult(audio, results)


