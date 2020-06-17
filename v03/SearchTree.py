from annoy import AnnoyIndex
from v02.Utils import *

# Read feature file
df = pd.read_csv('./feature/features-final-3.csv')
df.head()

# Convert features from df to list of list features
copydf = df.copy()
del copydf['filename']
features = copydf.values.tolist()
features = normalize_minmax_matrix(features)

# Get number of features
feature_nums = len(df.columns) - 1

# add feature data to annoy indexing
f = feature_nums
t = AnnoyIndex(f, metric='euclidean')

print(len(features))
for i in range(len(features)):
    v = features[i]
    t.add_item(i, v)


# create index tree
t.build(f)
t.save('./indextree/features-minmax-final-3.ann')

print("DONE")

print(features[1])