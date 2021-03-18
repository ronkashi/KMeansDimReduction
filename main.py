import argparse
from os.path import join
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, fetch_olivetti_faces, fetch_openml
from sklearn.metrics import normalized_mutual_info_score

from dim_reduction_methods import get_dim_reduction_transformer_dict
from metrics import sum_squared_norm_from_centroids, get_accuracy


def get_data(data_name):
    if data_name == 'SYNTH':
        return make_blobs(n_samples=1000, n_features=2000, centers=5, cluster_std=600,
                          center_box=(-1000, 1000), shuffle=True)
    elif data_name == 'ORL':
        return fetch_olivetti_faces(return_X_y=True)
    elif data_name == 'USPS':
        data_set = fetch_openml(data_id=41082)  # https://www.openml.org/d/41082
        return np.array(data_set['data']), np.array(data_set['target'], dtype=np.uint8)
    elif data_name in ['COIL20', 'PIE']:
        data_set = loadmat(join('data', data_name + '.mat'))
        return data_set['X'], data_set['Y'].ravel()
    else:
        return None


def plot_df(data_set_name: str, df: pd.DataFrame):
    import matplotlib.pyplot as plt
    for metric in ['Objective value', 'Accuracy', 'Running time', 'normalized_mutual_info_score']:
        f, ax = plt.subplots()
        for key, grp in df.groupby(['dim_reduction_method']):
            grp.plot(ax=ax, y=metric, label=key)
        plt.grid()
        plt.title(f"{data_set_name} : {metric} vs. number of dimensions (r)")
        plt.ylabel(metric)
        plt.show()


def produce_fit(kmeans_alg, features: np.ndarray, transformer: TransformerMixin):
    start_time = perf_counter()
    # dim reduction
    features_transformed = transformer.fit_transform(features)
    # K-means
    return kmeans_alg.fit_predict(features_transformed), perf_counter() - start_time


def run(simulation_args):
    # Data producer
    ds_features, targets = get_data(simulation_args.data_set)
    num_clusters = len(np.unique(targets))
    # K-Means constructor
    kmeans_alg = KMeans(n_clusters=num_clusters, n_init=5, max_iter=500)

    row_list = list()
    for r in range(5, 105, 5):
        for method_name, transformer in get_dim_reduction_transformer_dict(r, num_clusters).items():
            labels, running_time = produce_fit(kmeans_alg, ds_features, transformer)
            row_list.append([method_name, r, sum_squared_norm_from_centroids(ds_features, labels),
                             get_accuracy(labels, targets), running_time,
                             normalized_mutual_info_score(targets, labels)])

    df = pd.DataFrame(row_list, columns=['dim_reduction_method', 'r', 'Objective value', 'Accuracy', 'Running time',
                                         'normalized_mutual_info_score'])
    df.set_index('r', inplace=True)
    plot_df(simulation_args.data_set, df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-set', type=str, default='ORL', choices=['SYNTH', 'USPS', 'COIL20', 'ORL', 'PIE'])
    parsed_args = parser.parse_args()
    run(parsed_args)
