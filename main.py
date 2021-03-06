import argparse
import sklearn
import sklearn.datasets
from sklearn.random_projection import GaussianRandomProjection

from sklearn.cluster import KMeans
import numpy as np

def sum_squared_norm_from_centroids(data_points, labels):
    # TODO add here the math declaration of \mathbb{F} objective
    sum_squared = 0
    for label in np.unique(labels):
        points_of_labels = data_points[label == labels]
        sum_squared += (points_of_labels.shape[0] * np.sum(np.linalg.norm(points_of_labels, axis=1)**2) - np.linalg.norm(np.sum(points_of_labels, axis=0))**2)/points_of_labels.shape[0]
    return sum_squared / (np.linalg.norm(data_points)**2)


def get_dim_reduction_transformer(trans_name, r) -> sklearn.base.TransformerMixin:
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
        "Randomized Sampling with Exact SVD": None,
        "Randomized Sampling with Approximate SVD": None,
        "Random Projections": GaussianRandomProjection(r),
        "SVD": sklearn.decomposition.TruncatedSVD(r),
        "Approximate SVD": None,
        "Laplacian Scores": None
    }
    return trans_dict[trans_name]



def run(args):
    ### Data producer
    X = sklearn.datasets.fetch_olivetti_faces().data #ORL #TODO raplece with get dataset function

    for args.r in range(5,105,5):
        ### dim reduction
        if args.use_reduction:
            transformer = get_dim_reduction_transformer("SVD", args.r)
            X_red = transformer.fit_transform(X)
            # print(X_red.shape)
        else:
            X_red = X

        ### K-means
        # retreive 1. centroids, 2. labels -> calc relattive objective func F/\norm(A)_F^2
        kmeans_alg = KMeans(n_clusters=args.k, n_init=5, max_iter=500)
        kmeans_res = kmeans_alg.fit_predict(X_red)
        print(args.r, sum_squared_norm_from_centroids(X, kmeans_res))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=40)
    parser.add_argument('-use-reduction', type=bool, default=True)
    parser.add_argument('-r', type=int, default=20)
    parsed_args = parser.parse_args()
    run(parsed_args)
