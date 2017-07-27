import numpy as np

from classifiers import bayesian as by
from classifiers import pca
from classifiers import performance as pf
from histogram import histogram as hg
from mnist import read_mnist_data as mn
from prepare_data import classification_problems as cp
from statistics import basic as bs
from plots import plots as plt

outputExcelFile = r"Assignment_3 - Output.xlsx"

num_of_bins = 25
num_of_principal_components = 2
positive_number = 9
negative_number = 6
mnist_data = mn.get_mnist_data_for_numbers([positive_number, negative_number])
mean_vector, Z, C, eigenvalues, V, Vpc, P, pca_data, R, Xrec = pca.principal_component_analysis(data=mnist_data,
                                                                                                num_of_pcs=num_of_principal_components)
print('pca_data: ', pca_data)
print('P: ', P)
labeled_data = cp.reformat_data(pca_data)
plt.scatter_plot(P, pca_data['class_labels'], positive_number, negative_number)

print('mnist data size: ', len(mnist_data['feature_vectors']), len(mnist_data['class_labels']))
print('pca data size: ', len(pca_data['feature_vectors']['feature0']), len(pca_data['class_labels']))
print('class data size', {d[0]: len(d[1]) for d in labeled_data.items()})
# print('eigenvalues: ', eigenvalues)

class_mean_vectors = {d[0]: bs.mean_vector(d[1]) for d in labeled_data.items()}
class_covariances = {d[0]: bs.covariance_matrix(d[1]) for d in labeled_data.items()}

print('class_mean_vectors: ', class_mean_vectors)
print('class_covariances: ', class_covariances)

histogram = hg.create_histogram(pca_data, num_of_bins)
bayesian_histogram = hg.create_histogram_from_bayesian_params(pca_data, num_of_bins)

first_positive_index = next(i for i, num in enumerate(mnist_data['class_labels']) if num == positive_number)
first_negative_index = next(i for i, num in enumerate(mnist_data['class_labels']) if num == negative_number)

positive_representation = {
    'feature_vectors': [mnist_data['feature_vectors'][first_positive_index]],
    'class_labels': [mnist_data['class_labels'][first_positive_index]]
}

negative_representation = {
    'feature_vectors': [mnist_data['feature_vectors'][first_negative_index]],
    'class_labels': [mnist_data['class_labels'][first_negative_index]]
}

print('positive class label: ', positive_representation['class_labels'])
print('negative class label: ', negative_representation['class_labels'])

# plt.vector_to_image(28, 1, positive_representation['feature_vectors'][0])
mn.show(positive_representation['feature_vectors'][0])
mn.show(negative_representation['feature_vectors'][0])

mvp, Zp, Cp, evp, Vp, Vpcp, Pp, pdp, Rp, Xrecp = pca.principal_component_analysis(data=positive_representation,
                                                                                  mean_vector_force=mean_vector,
                                                                                  covariance_force=C,
                                                                                  num_of_pcs=num_of_principal_components)
X_for_classification_p = np.column_stack(pdp['feature_vectors'].values())[0]

mvn, Zn, Cn, evn, Vn, Vpcn, Pn, pdn, Rn, Xrecn = pca.principal_component_analysis(data=negative_representation,
                                                                                  mean_vector_force=mean_vector,
                                                                                  covariance_force=C,
                                                                                  num_of_pcs=num_of_principal_components)
X_for_classification_n = np.column_stack(pdn['feature_vectors'].values())[0]

mn.show(mvp.reshape((28, 28)))
mn.show(mvn.reshape((28, 28)))

mn.show(Cp)
mn.show(Cn)

mn.show(Vpcp[0].reshape((28, 28)))
mn.show(Vpcp[1].reshape((28, 28)))

mn.show(Vpcn[0].T.reshape((28, 28)))
mn.show(Vpcn[1].T.reshape((28, 28)))

mn.show(Rp.reshape((28, 28)))
mn.show(Rn.reshape((28, 28)))

mn.show(Xrecp.reshape((28, 28)))
mn.show(Xrecn.reshape((28, 28)))

mn.show(histogram[positive_number])
mn.show(histogram[negative_number])

mn.show(bayesian_histogram[positive_number])
mn.show(bayesian_histogram[negative_number])

# plt.vector_to_image(28, 1, Xrecp.reshape((28, 28)))

print('X after pca for positive: ', X_for_classification_p)
print('X after pca for negative: ', X_for_classification_n)

print('histogram classification for positive: ',
      by.histogram_classifier_for_target(pca_data, num_of_bins, histogram, X_for_classification_p, positive_number))
print('histogram classification for negative: ',
      by.histogram_classifier_for_target(pca_data, num_of_bins, histogram, X_for_classification_n, negative_number))

print('bayesian classification for positive: ',
      by.bayesian_classifier_for_target(pca_data, X_for_classification_p, positive_number))
print('bayesian classification for negative: ',
      by.bayesian_classifier_for_target(pca_data, X_for_classification_n, negative_number))

# print(pf.evaluate_training_accuracy(pca_data, labeled_data, histogram, num_of_bins, positive_number, negative_number))

# eo.writeExcelData(data=[{index: item for index, item in enumerate(mean_vector)}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=2,
#                   startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(V[0])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=3,
#                   startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(V[1])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=4,
#                   startCol=2)
#
# eo.writeExcelData(data={index: item for index, item in enumerate(histogram[positive_number])},
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=20,
#                   startCol=2)
# eo.writeExcelData(data={index: item for index, item in enumerate(histogram[negative_number])},
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=46,
#                   startCol=2)
#
# eo.writeExcelData(
#     data=[{index: item for index, item in enumerate(positive_representation['feature_vectors'][0].flatten())}],
#     excelFile=outputExcelFile,
#     sheetName='Results',
#     startRow=74,
#     startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(Zp[0])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=75,
#                   startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(pdp['feature_vectors']['feature0'])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=76,
#                   startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(pdp['feature_vectors']['feature1'])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=76,
#                   startCol=3)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(Rp[0])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=77,
#                   startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(Xrecp[0])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=78,
#                   startCol=2)
#
# eo.writeExcelData(
#     data=[{index: item for index, item in enumerate(negative_representation['feature_vectors'][0].flatten())}],
#     excelFile=outputExcelFile,
#     sheetName='Results',
#     startRow=80,
#     startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(Zn[0])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=81,
#                   startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(pdn['feature_vectors']['feature0'])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=82,
#                   startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(pdn['feature_vectors']['feature1'])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=82,
#                   startCol=3)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(Rn[0])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=83,
#                   startCol=2)
# eo.writeExcelData(data=[{index: item for index, item in enumerate(Xrecn[0])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=84,
#                   startCol=2)
