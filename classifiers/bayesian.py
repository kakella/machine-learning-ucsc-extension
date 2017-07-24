import math

import numpy as np

from histogram import histogram as hg
from prepare_data import classification_problems as cp
from statistics import basic as bs

indeterminate_value = "Indeterminate"


def probability_density_function_1d(input_value, mean, stddev):
    """
    This function calculates the PDF value for a 1-dimensional distribution

    :param input_value: The input value for which PDF value needs to be calculated
    :param mean: Mean of the distribution
    :param stddev: Standard deviation of the distribution
    :return: The PDF value for the input value
    """
    return 1 / (math.sqrt(2 * math.pi) * stddev) * (math.e ** (-1 / 2 * (((input_value - mean) / stddev) ** 2)))


def probability_density_function_nd(input_feature_vector, mean_vector, covariance):
    """
    This function calculates the PDF value for a 1-dimensional distribution

    :param input_feature_vector: The input value for which PDF value needs to be calculated
    :param mean_vector: Mean of the distribution
    :param covariance: Covariance of the distribution
    :return: The PDF value for the input feature vector
    """
    diff = input_feature_vector - mean_vector
    product = np.dot(diff, np.linalg.inv(covariance))
    product = np.dot(product, diff.T)
    return 1 / (
        (2 * math.pi) ** (len(input_feature_vector) / 2) *
        math.sqrt(bs.matrix_determinant(covariance))
    ) * (math.e ** (-0.5 * product))


def bayesian_1d(input_value, mean, stddev, num_of_samples):
    """
    This function calculates the Bayesian equivalent histogram value for the given input value, using the PDF .
    This function is valid for 1-dimensional data.

    :param input_value: The input value for which equivalent histogram value needs to be calculated
    :param mean: Mean of the distribution
    :param stddev: Standard deviation of the distribution
    :param num_of_samples: The number of samples in the distribution
    :return: The Bayesian equivalent histogram value for the given input value
    """
    return num_of_samples * probability_density_function_1d(input_value, mean, stddev)


def bayesian_nd(input_feature_vector, mean_vector, covariance, num_of_samples):
    """
    This function calculates the Bayesian equivalent histogram value for the given input value, using the PDF function.
    This function is valid for n-dimensional data.

    :param input_feature_vector: The input value for which equivalent histogram value needs to be calculated
    :param mean_vector: Mean of the distribution
    :param covariance: Covariance of deviation of the distribution
    :param num_of_samples: The number of samples in the distribution
    :return: The Bayesian equivalent histogram value for the given input value
    """
    return num_of_samples * probability_density_function_nd(input_feature_vector, mean_vector, covariance)


def histogram_classifier(data, num_of_bins, histogram, query_feature_vector, min_max_values_force=None):
    """
    This function finds the appropriate bin for query feature vector in the n-dimensional histogram,
    and then returns the bin values for every class label

    :param data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :param num_of_bins: Num of bins in the histogram (across all features)
    :param histogram: A dictionary of class label -> ndarray representing the histogram for each unique class label
        The ndarray has as many dimensions as the number of features in the input data
    :param query_feature_vector: The query vector for which we want to find bin number
    :param min_max_values_force: Force specific min, max values for each feature instead of calculating them.
        This is a list of tuples of the form [(feature1_min, feature1_max), (feature2_min, feature2_max), ...]
    :return: A dictionary of class labels -> bin values corresponding to the query feature vector
    """
    bin_number = hg.query_bin_number(data, num_of_bins, query_feature_vector, min_max_values_force)
    return {h[0]: h[1][tuple(bin_number)] for h in histogram.items()}


def histogram_classifier_for_target(data, num_of_bins, histogram, query_feature_vector,
                                    target_class_label, min_max_values_force=None):
    """
    This function finds the appropriate bin for query feature vector in the n-dimensional histogram,
    and then returns the probability (as a percentage) of that feature vector for the target class label

    :param data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :param num_of_bins: Num of bins in the histogram (across all features)
    :param histogram: A dictionary of class label -> ndarray representing the histogram for each unique class label
        The ndarray has as many dimensions as the number of features in the input data
    :param query_feature_vector: The query vector for which we want to find bin number
    :param target_class_label: The target class label
    :param min_max_values_force: Force specific min, max values for each feature instead of calculating them.
        This is a list of tuples of the form [(feature1_min, feature1_max), (feature2_min, feature2_max), ...]
    :return: The probability (as a percentage) of that feature vector for the target class label,
        or the string 'Indeterminate' if there is insufficient data to make a prediction
    """
    bin_data = histogram_classifier(data, num_of_bins, histogram, query_feature_vector, min_max_values_force)
    total = sum(bin_data.values())
    if total == 0:
        return indeterminate_value
    else:
        return bin_data[target_class_label] / total * 100


def prepare_data_for_bayesian_classifier(data):
    labeled_data = cp.reformat_data(data)
    num_of_features = len(labeled_data)

    mean_vectors = {}
    covariances = {}

    for cl in labeled_data.items():
        if num_of_features > 1:
            mean_vectors[cl[0]] = bs.mean_vector(cl[1])
            covariances[cl[0]] = bs.covariance_matrix(cl[1])
        else:
            flattened_data = np.array(cl[1].flatten(), dtype=float)
            mean_vectors[cl[0]] = bs.average(flattened_data)
            covariances[cl[0]] = bs.stddev(flattened_data)

    return labeled_data, mean_vectors, covariances


def bayesian_classifier(labeled_data, mean_vectors, covariances, query_feature_vector):
    """
    This function finds the appropriate Bayesian classifier value for query feature vector in the
    n-dimensional data for every class label

    :param labeled_data: TODO
    :param query_feature_vector: The query vector for which we want to find bin number. In case of 1-dimensional data,
        this value will be a scalar
    :return: A dictionary of class labels -> Bayesian classifier values corresponding to the query feature vector
    """

    bayesian_values = {}
    num_of_features = len(labeled_data)

    for cl in labeled_data.items():
        if num_of_features > 1:
            bayesian_values[cl[0]] = bayesian_nd(query_feature_vector,
                                                 mean_vectors[cl[0]],
                                                 covariances[cl[0]],
                                                 len(cl[1]))
        else:
            flattened_data = np.array(cl[1].flatten(), dtype=float)
            bayesian_values[cl[0]] = bayesian_1d(query_feature_vector,
                                                 mean_vectors[cl[0]],
                                                 covariances[cl[0]],
                                                 len(flattened_data))

    return bayesian_values


def bayesian_classifier_for_target(data, query_feature_vector, target_class_label):
    """
    This function finds the appropriate bin for query feature vector in the Gaussian distribution,
    and then returns the probability (as a percentage) of that feature vector for the target class label

    :param data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :param query_feature_vector: The query vector for which we want to find bin number
    :param target_class_label: The target class label for which the
    :return: The probability (as a percentage) of that feature vector for the target class label,
        or the string 'Indeterminate' if there is insufficient data to make a prediction
    """
    labeled_data, mean_vectors, covariances = prepare_data_for_bayesian_classifier(data)
    bayesian_values = bayesian_classifier(labeled_data, mean_vectors, covariances, query_feature_vector)
    total = sum(bayesian_values.values())
    if total == 0:
        return indeterminate_value
    else:
        return bayesian_values[target_class_label] / total * 100
