from v03.Utils import *
from collections import Counter

def euclidian_distance(instance1, instance2):
    result = np.sum(np.power(instance1 - instance2, 2))
    return np.sqrt(result)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
y1, sr1 = loadAudio('../train/agreement_1.wav')
y2, sr2 = loadAudio('../train/agreement_en(4).wav')
ect = euclidian_distance(a, b)
ect2 = euclidian_distance(get_feature_vector(y1, sr1), get_feature_vector(y2, sr2))

def get_neighbors(training_set,
                  labels,
                  test_instance,
                  k,
                  distance=euclidian_distance):
    """
    get_neighors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The list neighbors contains 3-tuples with
    (index, dist, label)
    where
    index    is the index from the training_set,
    dist     is the distance between the test_instance and the
             instance training_set[index]
    distance is a reference to a function used to calculate the
             distances
    """
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors


def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[getAudioName(neighbor[2])] += 1
    return class_counter.most_common(1)[0][0]