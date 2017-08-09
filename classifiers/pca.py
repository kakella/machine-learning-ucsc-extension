import numpy as np
import numpy.linalg as nl

from statistics import basic as bs


def principal_component_analysis(data, mean_vector_force=None, covariance_force=None, num_of_pcs=-1, datatype='unflattened'):
    if datatype == 'unflattened':
        X = np.array([fv.flatten() for fv in data['feature_vectors']])
        T = data['class_labels']
    else:
        X = data
        T = []

    if num_of_pcs == -1:
        num_of_pcs = len(X[0])

    print('len of X: ', len(X), ' width of X: ', len(X[0]))
    print('len of T: ', len(T))

    if mean_vector_force is not None:
        mean_vector = mean_vector_force
    else:
        mean_vector = bs.mean_vector(X)

    print('len of mu: ', len(mean_vector))

    Z = X - mean_vector
    print('len of Z: ', len(Z), ' width of Z: ', len(Z[0]))

    if covariance_force is not None:
        C = covariance_force
    else:
        C = np.cov(Z, rowvar=False)
        # C = bs.covariance_matrix(Z)

    print('len of C: ', len(C), ' width of C: ', len(C[0]))

    eigenvalues, V = nl.eigh(C)
    eigenvalues = np.flipud(eigenvalues)
    V = np.flipud(V.T)
    print('len of V: ', len(V), ' width of V: ', len(V[0]))

    Vpc = V[:num_of_pcs]
    # print('eigenvectors: ', Vpc)
    print('len of Vpc: ', len(Vpc), ' width of Vpc: ', len(Vpc[0]))
    # print('norm check of Vpc: ', np.linalg.norm(Vpc[0]), np.linalg.norm(Vpc[1]))
    # print('orthogonality check of Vpc: ', np.dot(Vpc[0, :], Vpc[1, :]))

    P = np.dot(Z, Vpc.T)
    print('len of P: ', len(P), ' width of P: ', len(P[0]))
    print('mean check of P: ', np.mean(P, axis=0))

    R = np.dot(P, Vpc)
    print('len of R: ', len(R), ' width of R: ', len(R[0]))

    Xrec = R + mean_vector
    print('len of Xrec: ', len(Xrec), ' width of Xrec: ', len(Xrec[0]))

    return mean_vector, \
           Z, \
           C, \
           eigenvalues, \
           V, \
           Vpc, \
           P, \
           {
               'feature_vectors': {'feature' + str(i): P[:, i] for i in range(num_of_pcs)},
               'class_labels': T
           }, \
           R, \
           Xrec
