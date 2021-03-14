import argparse
import sklearn
import sklearn.pipeline
import sklearn.datasets
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from time import perf_counter


def sum_squared_norm_from_centroids(data_points, labels):
    # TODO add here the math declaration of \mathbb{F} objective
    sum_squared = 0
    for label in np.unique(labels):
        points_of_labels = data_points[label == labels]
        sum_squared += (points_of_labels.shape[0] * np.sum(
            np.linalg.norm(points_of_labels, axis=1) ** 2) - np.linalg.norm(np.sum(points_of_labels, axis=0)) ** 2) / \
                       points_of_labels.shape[0]
    return sum_squared / (np.linalg.norm(data_points) ** 2)


def get_dim_reduction_transformer(trans_name, r, n_features_z_matrix: int = 0, eps=1/3) -> sklearn.base.TransformerMixin:
    """
    1. Randomized Sampling with Exact SVD (Sampl/SVD). This corresponds to Algorithm 1 with the following modification.
    In the first step of the algorithm, the matrix Z is calculated to contain exactly the
    top k right singular vectors of A.
    2. Randomized Sampling with Approximate SVD (Sampl/ApproxSVD). This corresponds to Algorithm 1
    with ε fixed to 1/3.
    3. Random Projections (RP). Here we use Algorithm 2. However, in our implementation we use the naive
    approach for the matrix-matrix multiplication in the third step (not the Mailman algorithm [25]).
    4. SVD. This is Algorithm 3 with the following modification. In the first step of the algorithm, the matrix
    Z is calculated to contain exactly the top k right singular vectors of A.
    5. Approximate SVD (ApprSVD). This corresponds to Algorithm 3 with ε fixed to 1/3.
    6. Laplacian Scores (LapScores).
    """
    trans_dict = {
        "Randomized Sampling with Exact SVD":
            sklearn.preprocessing.FunctionTransformer(rand_sampling_with_exact_svd,
                                                      kw_args={"r": r,
                                                               "n_features_z_matrix": n_features_z_matrix}),
        "Randomized Sampling with Approximate SVD":
            sklearn.preprocessing.FunctionTransformer(rand_sampling_with_exact_svd,
                                                      kw_args={"r": r,
                                                               "n_features_z_matrix": n_features_z_matrix,
                                                               'approx_svd': True,
                                                               'eps': 1/3}),
        "Random Projections": GaussianRandomProjection(r),
        "SVD": sklearn.decomposition.TruncatedSVD(r),
        "Approximate SVD": sklearn.preprocessing.FunctionTransformer(approximate_svd, kw_args={"k": r, "eps": eps}),
        # "Laplacian Scores": None,
        "K-Means": sklearn.preprocessing.FunctionTransformer()  # identity transformer
    }
    return trans_dict[trans_name]


def rand_sampling_with_exact_svd(mat: np.ndarray, n_features_z_matrix, r, approx_svd=False, eps=0):
    if approx_svd:
        z_mat = fast_forbenius_svd(mat, n_features_z_matrix, eps)
    else:
        z_mat = get_right_svd_decomposion_truncated(mat, n_features_z_matrix)
    columns_ind_sampled, scaling_factors = randomize_sampling(z_mat, r)
    return mat[:, columns_ind_sampled] * scaling_factors


def get_right_svd_decomposion_truncated(mat: np.ndarray, k):
    # TODO this function is not optimized and have redundant calculations
    u, s, vh = np.linalg.svd(mat, compute_uv=True, full_matrices=False)
    return np.transpose(vh[:k])


def fast_forbenius_svd(mat: np.ndarray, k: int, eps):
    y_mat = GaussianRandomProjection(k + int(k / eps) + 1).fit_transform(mat)
    q_mat, _ = np.linalg.qr(y_mat)
    return get_right_svd_decomposion_truncated(np.matmul(np.transpose(q_mat), mat), k)


def approximate_svd(mat: np.ndarray, k: int, eps):
    return np.matmul(mat, fast_forbenius_svd(mat, k, eps))


def randomize_sampling(mat: np.ndarray, r) -> np.ndarray:
    # Calculation of the p_i (TODO latex)
    probabilities = np.linalg.norm(mat, axis=0) ** 2 / (np.linalg.norm(mat) ** 2)
    # Sampling columns indexes according to the p_i's
    columns_ind_sampled = np.random.choice(probabilities.size, r, p=probabilities)
    # Rescaling columns by 1/√(r * p_i)
    return columns_ind_sampled, np.sqrt(r * probabilities[columns_ind_sampled])
    # return mat[:, columns_ind_sampled] / np.sqrt(r * probabilities[columns_ind_sampled])


def get_accuracy(pred_labels, true_labels):
    num_pred_mistakes = 0
    for lbl in np.unique(pred_labels):
        num_pred_mistakes += np.sum(np.bincount(true_labels[pred_labels == lbl])) - np.max(
            np.bincount(true_labels[pred_labels == lbl]))
    return 1 - num_pred_mistakes / true_labels.size


def get_data(data_name):
    if data_name == 'SYNTH':
        return sklearn.datasets.make_blobs(n_samples=1000, n_features=2000, centers=5, cluster_std=600,
                                           center_box=(-1000, 1000), shuffle=True)
    elif data_name == 'ORL':
        data_set = sklearn.datasets.fetch_olivetti_faces(shuffle=True)
        return data_set.data, data_set.target
    else:
        return None


def run(args):
    ### Data producer
    ds_features, targets = get_data(args.data_set)

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
    plot_df(args.data_set, df)


def plot_df(data_set_name: str, df: pd.DataFrame):
    import matplotlib.pyplot as plt
    for metric in ['Objective value', 'Accuracy', 'Running time']:
        f, ax = plt.subplots()
        for key, grp in df.groupby(['trans_name']):
            grp.plot(ax=ax, y=metric, label=key)
        plt.grid()
        plt.title(f"{data_set_name} : {metric} vs. number of dimmenstions (r)")
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
    parser.add_argument('--data-set', type=str, default='ORL')
    # parser.add_argument('-use-reduction', type=bool, default=True)
    # parser.add_argument('-r', type=int, default=20)
    parsed_args = parser.parse_args()
    run(parsed_args)
