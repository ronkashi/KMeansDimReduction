import numpy as np


def sum_squared_norm_from_centroids(data_points, labels):
    # TODO add here the math declaration of \math{F} objective
    sum_squared = 0
    for label in np.unique(labels):
        points_of_labels = data_points[label == labels]
        sum_squared += (points_of_labels.shape[0] * np.sum(np.linalg.norm(points_of_labels, axis=1) ** 2) -
                        np.linalg.norm(np.sum(points_of_labels, axis=0)) ** 2) / \
                       points_of_labels.shape[0]
    return sum_squared / (np.linalg.norm(data_points) ** 2)


def get_accuracy(pred_labels, true_labels):
    num_pred_mistakes = 0
    for lbl in np.unique(pred_labels):
        num_pred_mistakes += np.sum(np.bincount(true_labels[pred_labels == lbl])) - \
                             np.max(np.bincount(true_labels[pred_labels == lbl]))
    return 1 - num_pred_mistakes / true_labels.size
