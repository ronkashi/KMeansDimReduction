from typing import Tuple, Dict

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer
from sklearn.random_projection import GaussianRandomProjection

from laplacian_scores import construct_W, lap_score


def get_dim_reduction_transformer_dict(r, n_features_z_matrix: int = 0, eps=1 / 3) -> Dict[str, TransformerMixin]:
    """
    1. Randomized Sampling with Exact SVD (Sampled/SVD). This corresponds to Algorithm 1 with the following modification
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
    dimension_reduction_methods_dict = {
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
        "Laplacian Scores": FunctionTransformer(laplace_score, kw_args={"r": r}),
        "K-Means": FunctionTransformer()  # identity transformer
    }
    return dimension_reduction_methods_dict


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
    # Calculation of the p_i latex : $p_i = \frac{\norm{{X}_{(i)}}^2_2}{\norm{{X}}^2_F}$
    probabilities = np.linalg.norm(mat, axis=1) ** 2 / (np.linalg.norm(mat) ** 2)
    # Sampling columns indexes according to the p_i's
    columns_ind_sampled = np.random.choice(probabilities.size, r, p=probabilities)
    # Rescaling columns by 1/√(r * p_i)
    return columns_ind_sampled, np.sqrt(r * probabilities[columns_ind_sampled])


def laplace_score(mat: np.ndarray, r):
    kwargs_w = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    weights = construct_W.construct_W(mat, **kwargs_w)

    # obtain the scores of features
    score = lap_score.lap_score(mat, W=weights)
    # sort the feature scores in an ascending order according to the feature scores
    idx = lap_score.feature_ranking(score)
    return mat[:, idx[:r]]
