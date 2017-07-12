import math as mt

import numpy as np


def average(data):
    """
    This function returns the mean of the scalar data (single column and multiple rows)

    :param data: A list of scalar data points
    :return: The mean of the data
    """
    return sum(data) / len(data)


def variance(data):
    """
    This function returns the variance of the scalar data (single column and multiple rows)

    :param data: A list of scalar data points
    :return: The variance of the data, by dividing by len(data)-1 ie. considered as a sample population
    """
    sum_of_squares = 0
    mean = average(data)
    for d in data:
        sum_of_squares += ((d - mean) ** 2)
    return sum_of_squares / (len(data) - 1)


def stddev(data):
    """
    This function returns the standard deviation of the scalar data (single column and multiple rows)

    :param data: A list of scalar data points
    :return: The standard deviation of the data, by dividing by len(data)-1 ie. considered as a sample population
    """
    return mt.sqrt(variance(data))


def mean_vector(nd_data):
    """
    This function returns the mean vector for the 2-dimensional data (multiple columns and multiple rows)

    :param nd_data: A 2-dimensional list of data points
    :return: The mean vector of the data (list of means for each column of data)
    """
    mv = np.zeros(len(nd_data[0]))
    for d in nd_data:
        mv = mv + d
    mv /= len(nd_data)
    return mv


def mean_vector_training_set(nd_data):
    """
    This function returns the mean vector for the 2-dimensional data (multiple columns and multiple rows).
    The data is required to be arranged in the standard format for a classifier's training set.

    :param nd_data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :return: The mean vector of the data (list of means for each column of data)
    """
    return [average(f) for f in nd_data['feature_vectors'].values()]


def covariance_matrix(nd_data):
    """
    This function returns the covariance matrix for the 2-dimensional data (multiple columns and multiple rows).

    :param nd_data: A 2-dimensional list of data points
    :return: The covariance matrix of the data, by dividing by len(data)-1 ie. considered as a sample population
    """
    num_of_features = len(nd_data[0])
    data_size = len(nd_data)

    mean_vec = mean_vector(nd_data)
    covariance = np.zeros((num_of_features, num_of_features))

    for i in range(num_of_features):
        for j in range(num_of_features):
            for k in range(data_size):
                covariance[i][j] += (nd_data[k][i] - mean_vec[i]) * (nd_data[k][j] - mean_vec[j])
            covariance[i][j] /= (data_size - 1)

    return covariance


def covariance_matrix_training_set(nd_data):
    """
    This function returns the covariance matrix for the 2-dimensional data (multiple columns and multiple rows).
    The data is required to be arranged in the standard format for a classifier's training set.

    :param nd_data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :return: The covariance matrix of the data, by dividing by len(data)-1 ie. considered as a sample population
    """
    feature_list = list(nd_data['feature_vectors'].keys())
    num_of_features = len(feature_list)
    data_size = len(nd_data['class_labels'])

    mean_vec = mean_vector_training_set(nd_data)
    covariance = np.zeros((num_of_features, num_of_features))

    for i in range(num_of_features):
        n = feature_list[i]
        for j in range(num_of_features):
            m = feature_list[j]

            for k in range(data_size):
                covariance[i][j] += (nd_data['feature_vectors'][n][k] - mean_vec[i]) * \
                                    (nd_data['feature_vectors'][m][k] - mean_vec[j])
            covariance[i][j] /= (data_size - 1)

    return covariance


def matrix_determinant(matrix):
    """
    This function finds the determinant of a NxN matrix

    TODO: Implement this function on your own instead of using numpy function

    :param matrix: A NxN matrix of data points
    :return: The determinant of the matrix (a scalar value)
    """
    return np.linalg.det(matrix)
