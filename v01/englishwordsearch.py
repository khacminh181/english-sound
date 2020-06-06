from annoy import AnnoyIndex
from v01.Utils import *
from collections import Counter
import pandas as pd


df = pd.read_csv('../features/featuresplayground100-1.csv')
# List file name
words = df["filename"].tolist()

def get_audio_feature_vector(audio):
    y, sr = loadAudio(audio)
    feat = audio_2_feature_vector(y, sr)
    return np.array(feat)

def get_result(feat, words, searchK):
    results = []
    result = u.get_nns_by_vector(feat, n=searchK)
    result_songs = [words[k] for k in result]
    results.append(result_songs)
    print(results)

    return np.array(results).flatten()

def printResult(audio, results):
    print("Search for word {}".format(audio))
    print("RESULT")
    most_song = Counter(results)
    print(most_song.most_common())

# load index tree
f = len(df.columns) - 1
# u = AnnoyIndex(f, metric='angular')
# u.load('../indextree/englishwordfeatures.ann')
u = AnnoyIndex(f, metric='euclidean')
u.load('../indextree/englishwordfeatures-minmax100-1.ann')

# search

# audio = "../audio/minimal--_gb_1.wav"
# audio = "../audio/book.wav"
audio = "../audio-test/book--_us_1 (online-audio-converter.com).wav"
# audio = "../audio-test/ordinary--_us_1 (online-audio-converter.com).wav"
feat = get_audio_feature_vector(audio)
results = get_result(feat, words, 5)
printResult(audio, results)


