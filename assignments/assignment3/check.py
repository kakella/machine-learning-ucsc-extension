import numpy as np

from classifiers import pca
from mnist import read_mnist_data as mn
from prepare_data import classification_problems as cp
from statistics import basic as bs

num_of_bins = 25
num_of_principal_components = 2
mnist_data = mn.get_mnist_data_for_numbers([9, 6])
mean_vector, Z, C, eigenvalues, V, pca_data, R = pca.principal_component_analysis(data=mnist_data,
                                                                                  num_of_pcs=num_of_principal_components)
labeled_data = cp.reformat_data(pca_data)

print(np.min(Z[0]), np.max(Z[0]))
mean_of_Z = bs.mean_vector(Z)
print(min(mean_of_Z), max(mean_of_Z))
print(['False' for f in (C == C.T).flatten() if f is False])
print(np.linalg.norm(V[0]), np.linalg.norm(V[1]))
print(np.dot(V[0, :], V[1, :]))
print(bs.mean_vector_training_set(pca_data))
