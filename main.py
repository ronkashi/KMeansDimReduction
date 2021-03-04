import argparse
import os

import sklearn.datasets
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
import numpy as np

def sum_squared_norm_from_centroids(data_points, labels):
    sum_squared = 0
    for label in np.unique(labels):
        points_of_labels = data_points[label == labels]
        sum_squared += (points_of_labels.shape[0] * np.sum(np.linalg.norm(points_of_labels, axis=1)**2) - np.linalg.norm(np.sum(points_of_labels, axis=0))**2)/points_of_labels.shape[0]
    return sum_squared



def run(args):
    ### Data producer
    X = sklearn.datasets.fetch_olivetti_faces().data

    ### dim reduction
    if args.use_reduction:
        transformer = GaussianRandomProjection(args.r)
        X = transformer.fit_transform(X)
        print(X.shape)

    ### K-means
    # retreive 1. centroids, 2. labels -> calc relattive objective func F/\norm(A)_F^2
    kmeans_alg = KMeans(n_clusters=args.k, n_init=5, max_iter=500)
    kmeans_res = kmeans_alg.fit_predict(X)
    print(sum_squared_norm_from_centroids(X, kmeans_res))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=40)
    parser.add_argument('-use-reduction', type=bool, default=False)
    parser.add_argument('-r', type=int, default=20)
    parsed_args = parser.parse_args()
    run(parsed_args)
