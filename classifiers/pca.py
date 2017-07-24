import numpy as np
import numpy.linalg as nl

from statistics import basic as bs


def principal_component_analysis(data, mean_vector_force=None, covariance_force=None, num_of_pcs=-1):
    X = np.array([fv.flatten() for fv in data['feature_vectors']])
    T = data['class_labels']

    if mean_vector_force is not None:
        mean_vector = mean_vector_force
    else:
        mean_vector = bs.mean_vector(X)

    Z = X - mean_vector

    if covariance_force is not None:
        C = covariance_force
    else:
        C = np.cov(Z, rowvar=False)

    eigenvalues, V = nl.eigh(C)
    eigenvalues = np.flipud(eigenvalues)
    V = np.flipud(V.T)
    P = np.dot(Z, V.T)
    R = np.dot(P, V)

    if num_of_pcs == -1:
        num_of_pcs = len(X[0])

    return mean_vector, \
           Z, \
           C, \
           eigenvalues, \
           V, \
           {
               'feature_vectors': {'feature' + str(i): P[:, i] for i in range(num_of_pcs)},
               'class_labels': T
           }, \
           R
