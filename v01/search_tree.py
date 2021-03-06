from annoy import AnnoyIndex
import pandas as pd
from v01.Utils import *

# Read feature file
df = pd.read_csv('../features/featuresplayground100-1.csv')
df.head()

# Convert feature from df to list of list
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
t.save('../indextree/englishwordfeatures-minmax100-1.ann')

print("DONE")