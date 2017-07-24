import numpy as np

from classifiers import bayesian as by
from classifiers import pca
from histogram import histogram as hg
from mnist import read_mnist_data as mn
from prepare_data import classification_problems as cp
from statistics import basic as bs

outputExcelFile = r"Assignment_3 - Output.xlsx"

num_of_bins = 25
num_of_principal_components = 2
mnist_data = mn.get_mnist_data_for_numbers([9, 6])
mean_vector, Z, C, eigenvalues, V, pca_data, R = pca.principal_component_analysis(data=mnist_data,
                                                                                  num_of_pcs=num_of_principal_components)
labeled_data = cp.reformat_data(pca_data)

print('mnist data size: ', len(mnist_data['feature_vectors']), len(mnist_data['class_labels']))
print('pca data size: ', len(pca_data['feature_vectors']['feature0']), len(pca_data['class_labels']))
print('class data size', {d[0]: len(d[1]) for d in labeled_data.items()})

class_mean_vectors = {d[0]: bs.mean_vector(d[1]) for d in labeled_data.items()}
class_covariances = {d[0]: bs.covariance_matrix(d[1]) for d in labeled_data.items()}

print('class_mean_vectors: ', class_mean_vectors)
print('class_covariances: ', class_covariances)

histogram = hg.create_histogram(pca_data, num_of_bins)
bayesian_histogram = hg.create_histogram_from_bayesian_params(pca_data, num_of_bins)

positive_representation = {
    'feature_vectors': [mnist_data['feature_vectors'][0]],
    'class_labels': mnist_data['class_labels'][0]
}

negative_representation = {
    'feature_vectors': [mnist_data['feature_vectors'][1]],
    'class_labels': mnist_data['class_labels'][1]
}

print('positive class label: ', positive_representation['class_labels'])
print('negative class label: ', negative_representation['class_labels'])

mvp, Zp, Cp, evp, Vp, pdp, Rp = pca.principal_component_analysis(data=positive_representation,
                                                                 mean_vector_force=mean_vector,
                                                                 covariance_force=C,
                                                                 num_of_pcs=num_of_principal_components)
Xrecp = Rp + mean_vector
X_for_classification_p = np.column_stack(pdp['feature_vectors'].values())[0]

mvn, Zn, Cn, evn, Vn, pdn, Rn = pca.principal_component_analysis(data=negative_representation,
                                                                 mean_vector_force=mean_vector,
                                                                 covariance_force=C,
                                                                 num_of_pcs=num_of_principal_components)
Xrecn = Rn + mean_vector
X_for_classification_n = np.column_stack(pdn['feature_vectors'].values())[0]

print('X after pca for positive: ', X_for_classification_p)
print('X after pca for negative: ', X_for_classification_n)

print('histogram classification for positive: ',
      by.histogram_classifier_for_target(pca_data, num_of_bins, histogram, X_for_classification_p, 9))
print('histogram classification for negative: ',
      by.histogram_classifier_for_target(pca_data, num_of_bins, histogram, X_for_classification_n, 6))

print('bayesian classification for positive: ', by.bayesian_classifier_for_target(pca_data, X_for_classification_p, 9))
print('bayesian classification for negative: ', by.bayesian_classifier_for_target(pca_data, X_for_classification_n, 6))

# print(pf.evaluate_training_accuracy(pca_data, labeled_data, histogram, num_of_bins, 9, 6))

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
# eo.writeExcelData(data={index: item for index, item in enumerate(histogram[9])},
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=20,
#                   startCol=2)
# eo.writeExcelData(data={index: item for index, item in enumerate(histogram[6])},
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=46,
#                   startCol=2)
#
# eo.writeExcelData(data=[{index: item for index, item in enumerate(positive_representation['feature_vectors'][0].flatten())}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=74,
#                   startCol=2)
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
# eo.writeExcelData(data=[{index: item for index, item in enumerate(negative_representation['feature_vectors'][0].flatten())}],
#                   excelFile=outputExcelFile,
#                   sheetName='Results',
#                   startRow=80,
#                   startCol=2)
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
