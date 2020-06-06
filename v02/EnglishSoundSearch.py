from annoy import AnnoyIndex
from v02.Utils import *
from collections import Counter
import pandas as pd

df = pd.read_csv('./feature/featuresplayground100-v02.csv')
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
        result = [words[k] for k in result]
        results.append(result)
    return np.array(results).flatten()

def printResult(audio, results):
    print("Search for word {}".format(audio))
    print("RESULT")
    most_song = Counter(results)
    print(most_song.most_common())

# load index tree
f = len(df.columns) - 1
u = AnnoyIndex(f, metric='euclidean')
u.load('./indextree/englishwordfeatures-minmax100-v02.ann')

# search
# audio = "../audio/minimal--_gb_1.wav"
# audio = "../audio/book.wav"
# audio = "../audio-test/book--_us_1 (online-audio-converter.com).wav"
audio = "../audio-test/ordinary--_us_1 (online-audio-converter.com).wav"
feat = get_audio_feature_vector(audio)
results = get_result(feat, words, 5)
printResult(audio, results)


