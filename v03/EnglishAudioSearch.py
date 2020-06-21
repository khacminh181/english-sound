from v03.knn import *

df = pd.read_csv('./features/features6.csv')
df.head()

df = df[(df != 0).all(1)]
# Convert features from df to list of list features
copydf = df.copy()
del copydf['filename']
features = copydf.values.tolist()
# features = normalize_minmax_matrix(features)

# Get number of features
feature_nums = len(df.columns) - 1

# List file name
words = df["filename"].tolist()

def get_audio_feature_vector(audio):
    y, sr = loadAudio(audio)
    feat = get_feature_vector(y, sr)
    return np.array(feat)

def get_result(feat, words, searchK):
    results = []
    for vector_feat in feat:
        result = get_neighbors(features, words, vector_feat, searchK)
        result = getAudioName(vote(result))
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

# search
# audio = "../audio/minimal--_gb_1.wav"
# audio = "../audio/book.wav"
# audio = "../test/ordinary_en(1).wav"
# audio = "../test/pronunciation_en_swing (3).wav"
# audio = "../audio-test/ordinary--_us_1 (online-audio-converter.com).wav"
# audio = "../test/brick(10)_F.wav"

# audio = "../test/agreement_en(1).wav"
# audio = "../test/agreement_uk_M.wav"
# audio = "../test/international_en.wav"
# audio = "../test/international_F .wav"
# audio = "../test/kid_us_M(3).wav"
# audio = "../audio-test/book--_us_1 (online-audio-converter.com).wav"
audio = "../test/book_en(3).wav"
# audio = "../test/circle_en(1).wav"
# audio = "../test/minimal_en(1).wav"
# audio = "../audio-test/book--_us_1 (online-audio-converter.com).wav"

feat = get_audio_feature_vector(audio)
results = get_neighbors(features, words, feat, 10)
r = [getAudioName(i[2]) for i in results]
print(vote(results))



