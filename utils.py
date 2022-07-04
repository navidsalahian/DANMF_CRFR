
import numpy as np
from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, pairwise_distances
import pandas as pd

def normalize(X, n):

    Xmin = np.min(X, axis=1)
    Xmax = np.max(X, axis=1)
    mmin = np.matlib.repmat(Xmin.T, n, 1)
    mmax = np.matlib.repmat(Xmax.T, n, 1)
    Xu = X - mmin.T
    Xd = mmax.T - mmin.T
    X = Xu / np.maximum(Xd, 10**-10)
    return X.T

def preproccessing(args):

    print("\nPre-processing started.\n")
    data = io.loadmat(args.dataset_path)
    X = data['X']
    # get real labels
    Y = data['Y']
    d, n = X.shape
    near_n = NearestNeighbors(n_neighbors=args.k_neigh, metric='euclidean').fit(X.T)
    S = np.zeros((n, n))
    dist, indices = near_n.kneighbors(X.T)
    indices = indices[:, 1:]
    dist = dist[:, 1:]
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            S[i, indices[i, j]] = np.exp(-(dist[i, j] ** 2) / args.delta)  # Heat kernel similarity

    D = np.sum(S, axis=1)
    D = np.diag(D)
    L = D - S
    data = {'X': X, 'Y': Y, 'D': D, 'S': S, 'L': L}
    return data