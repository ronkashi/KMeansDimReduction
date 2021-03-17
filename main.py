import argparse
from time import perf_counter
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, fetch_olivetti_faces, fetch_openml
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer
from sklearn.random_projection import GaussianRandomProjection


def sum_squared_norm_from_centroids(data_points, labels):
    # TODO add here the math declaration of \math{F} objective
    sum_squared = 0
    for label in np.unique(labels):
        points_of_labels = data_points[label == labels]
        sum_squared += (points_of_labels.shape[0] * np.sum(np.linalg.norm(points_of_labels, axis=1) ** 2) -
                        np.linalg.norm(np.sum(points_of_labels, axis=0)) ** 2) / \
                       points_of_labels.shape[0]
    return sum_squared / (np.linalg.norm(data_points) ** 2)


def get_dim_reduction_transformer(trans_name, r, n_features_z_matrix: int = 0, eps=1 / 3) -> TransformerMixin:
    """
    1. Randomized Sampling with Exact SVD (Sampled/SVD). This corresponds to Algorithm 1 with the following modification.
    In the first step of the algorithm, the matrix Z is calculated to contain exactly the
    top k right singular vectors of A.
    2. Randomized Sampling with Approximate SVD (Sampled/ApproxSVD). This corresponds to Algorithm 1
    with ε fixed to 1/3.
    3. Random Projections (RP). Here we use Algorithm 2. However, in our implementation we use the naive
    approach for the matrix-matrix multiplication in the third step (not the Mailman algorithm [25]).
    4. SVD. This is Algorithm 3 with the following modification. In the first step of the algorithm, the matrix
    Z is calculated to contain exactly the top k right singular vectors of A.
    5. Approximate SVD (ApproximateSVD). This corresponds to Algorithm 3 with ε fixed to 1/3.
    6. Laplacian Scores (LapScores).
    """
    trans_dict = {
        "Randomized Sampling with Exact SVD":
            FunctionTransformer(rand_sampling_with_svd, kw_args={"r": r,
                                                                 "n_features_z_matrix": n_features_z_matrix}),
        "Randomized Sampling with Approximate SVD":
            FunctionTransformer(rand_sampling_with_svd, kw_args={"r": r,
                                                                 "n_features_z_matrix": n_features_z_matrix,
                                                                 'approx_svd': True,
                                                                 'eps': 1 / 3}),
        "Random Projections": GaussianRandomProjection(r),
        "SVD": TruncatedSVD(r),
        "Approximate SVD": FunctionTransformer(approximate_svd, kw_args={"k": r, "eps": eps}),
        "Laplacian Scores": None,
        "K-Means": FunctionTransformer()  # identity transformer
    }
    return trans_dict[trans_name]


def rand_sampling_with_svd(mat: np.ndarray, n_features_z_matrix, r, approx_svd=False, eps=0):
    if approx_svd:
        z_mat = fast_forbenius_svd(mat, n_features_z_matrix, eps)
    else:
        z_mat = get_right_svd_decomposition_truncated(mat, n_features_z_matrix)
    columns_ind_sampled, scaling_factors = randomize_sampling(z_mat, r)
    return mat[:, columns_ind_sampled] * scaling_factors


def get_right_svd_decomposition_truncated(mat: np.ndarray, k) -> np.ndarray:
    """
    Given matrix A the SVD decomposition A = U x S x transpose(V)
    With truncated SVD the output will be the k vectors with the greatest eigenvalues
    A_k = U_k x S_k x transpose(V_K)
    The code is a bit tricky part and based on:
    https://stackoverflow.com/questions/20681972/different-results-for-pca-truncated-svd-and-svds-on-numpy-and-sklearn
    :param mat: A. the input matrix to decompose
    :param k: the reduced dimension
    :return: the matrix V_k from the decomposition
    """
    return np.transpose(TruncatedSVD(k).fit(mat).components_)


def fast_forbenius_svd(mat: np.ndarray, k: int, eps):
    reduced_dim = min(mat.shape[1], k + int(k / eps) + 1)
    y_mat = GaussianRandomProjection(reduced_dim).fit_transform(mat)
    q_mat, _ = np.linalg.qr(y_mat)
    return get_right_svd_decomposition_truncated(np.matmul(np.transpose(q_mat), mat), k)


def approximate_svd(mat: np.ndarray, k: int, eps):
    return np.matmul(mat, fast_forbenius_svd(mat, k, eps))


def randomize_sampling(mat: np.ndarray, r) -> Tuple[np.ndarray, np.ndarray]:
    # Calculation of the p_i (TODO latex)
    probabilities = np.linalg.norm(mat, axis=1) ** 2 / (np.linalg.norm(mat) ** 2)
    # Sampling columns indexes according to the p_i's
    columns_ind_sampled = np.random.choice(probabilities.size, r, p=probabilities)
    # Rescaling columns by 1/√(r * p_i)
    return columns_ind_sampled, np.sqrt(r * probabilities[columns_ind_sampled])


def get_accuracy(pred_labels, true_labels):
    num_pred_mistakes = 0
    for lbl in np.unique(pred_labels):
        num_pred_mistakes += np.sum(np.bincount(true_labels[pred_labels == lbl])) - np.max(
            np.bincount(true_labels[pred_labels == lbl]))
    return 1 - num_pred_mistakes / true_labels.size


def get_data(data_name):
    if data_name == 'SYNTH':
        return make_blobs(n_samples=1000, n_features=2000, centers=5, cluster_std=600,
                          center_box=(-1000, 1000), shuffle=True)
    elif data_name == 'ORL':
        return fetch_olivetti_faces(return_X_y=True)
    elif data_name == 'USPS':
        data_set = fetch_openml(data_id=41082)  # https://www.openml.org/d/41082
        return np.array(data_set['data']), np.array(data_set['target'], dtype=np.uint8)
    else:
        return None


def run(simulation_args):
    # Data producer
    ds_features, targets = get_data(simulation_args.data_set)

    kmeans_alg = KMeans(n_clusters=len(np.unique(targets)), n_init=5, max_iter=500)
    row_list = list()
    for trans_name in ['Randomized Sampling with Exact SVD',
                       'Randomized Sampling with Approximate SVD',
                       'Random Projections', 'SVD', 'Approximate SVD', 'K-Means']:
        print(trans_name)
        for r in range(5, 105, 5):
            labels, running_time = produce_fit(kmeans_alg, ds_features, trans_name, r, len(np.unique(targets)))
            row_list.append([trans_name, r, sum_squared_norm_from_centroids(ds_features, labels),
                             get_accuracy(labels, targets), running_time])

    df = pd.DataFrame(row_list, columns=['trans_name', 'r', 'Objective value', 'Accuracy', 'Running time'])
    df.set_index('r', inplace=True)
    plot_df(simulation_args.data_set, df)


def plot_df(data_set_name: str, df: pd.DataFrame):
    import matplotlib.pyplot as plt
    for metric in ['Objective value', 'Accuracy', 'Running time']:
        f, ax = plt.subplots()
        for key, grp in df.groupby(['trans_name']):
            grp.plot(ax=ax, y=metric, label=key)
        plt.grid()
        plt.title(f"{data_set_name} : {metric} vs. number of dimensions (r)")
        plt.ylabel(metric)
        plt.show()


def produce_fit(kmeans_alg, features, trans_name: str, r: int, n_features_z_matrix: int):
    # dim reduction
    transformer = get_dim_reduction_transformer(trans_name, r, n_features_z_matrix)
    start_time = perf_counter()
    features_transformed = transformer.fit_transform(features)

    # K-means
    return kmeans_alg.fit_predict(features_transformed), perf_counter() - start_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-set', type=str, default='USPS', choices=['ORL', 'SYNTH', 'USPS'])
    parsed_args = parser.parse_args()
    run(parsed_args)
