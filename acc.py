import scipy
import numpy as np
from sklearn.metrics import accuracy_score


def cluster_acc(real_labels, labels):
    permutation = []
    n_clusters = len(np.unique(real_labels))
    labels = np.unique(labels, return_inverse=True)[1]
    for i in range(n_clusters):
        idx = labels == i
        if np.sum(idx) != 0:
            new_label = scipy.stats.mode(real_labels[idx])[0][0]
            permutation.append(new_label)
    new_labels = [permutation[label] for label in labels]
    return accuracy_score(real_labels, new_labels)



