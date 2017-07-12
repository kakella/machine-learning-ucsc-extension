import itertools as it

import numpy as np

from classifiers import bayesian as by


def create_bins(min_value, max_value, num_of_bins, data_type):
    """
    This function returns a list of the bin center values for creating histograms

    :param min_value: The smallest value in the data
    :param max_value: The largest value in the data
    :param num_of_bins: The number of bins in the histogram
    :param data_type: Type of bin values
    :return: A list of num_of_bins bin center values, and
             step: A float representing step size
    """
    bins_by_start_value, bin_size = np.linspace(start=min_value,
                                                stop=max_value,
                                                num=num_of_bins,
                                                retstep=True,
                                                dtype=data_type,
                                                endpoint=False)
    return bins_by_start_value + (bin_size / 2), bin_size


def assign_bin_for_value(value, min_value, max_value, num_of_bins):
    """
    This function returns the bin number for a value for building the histogram

    :param value: The value to evaluate
    :param min_value: The smallest value in the data
    :param max_value: The largest value in the data
    :param num_of_bins: The number of bins in the histogram
    :return: A bin number from 0 to num_of_bins-1
    """
    return int(round((num_of_bins - 1) * (value - min_value) / (max_value - min_value)))


def create_histogram(data, num_of_bins, min_max_values_force=None):
    """
    This function creates a histogram

    :param data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :param num_of_bins: Num of bins in the histogram (across all features)
    :param min_max_values_force: Force specific min, max values for each feature instead of calculating them.
        This is a list of tuples of the form [(feature1_min, feature1_max), (feature2_min, feature2_max), ...]
    :return: A dictionary of class label -> ndarray representing the histogram for each unique class label
        The ndarray has as many dimensions as the number of features in the input data
    """
    num_of_features = len(data['feature_vectors'])
    unique_class_labels = set(data['class_labels'])
    data_size = len(data['class_labels'])

    if min_max_values_force is None:
        min_max_values = [(min(f), max(f)) for f in data['feature_vectors'].values()]
    else:
        min_max_values = min_max_values_force
    print('min max values: ', min_max_values)

    print(data['feature_vectors'].keys())

    histogram = {cl: np.zeros(tuple([num_of_bins for _ in range(num_of_features)]))
                 for cl in unique_class_labels}

    for i in range(data_size):
        class_label = data['class_labels'][i]
        bin_number = np.zeros(num_of_features, dtype=np.int32)

        for f in range(num_of_features):
            value = data['feature_vectors']['feature' + str(f)][i]
            min_value = min_max_values[f][0]
            max_value = min_max_values[f][1]
            bin_number[f] = assign_bin_for_value(value, min_value, max_value, num_of_bins)

        histogram[class_label][tuple(bin_number)] += 1

    return histogram


def create_histogram_from_bayesian_params(data, num_of_bins):
    """
    This function creates a histogram from the appropriate Bayesian parameters. It is assumed that the distribution of
    data is normal (Gaussian)

    :param data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :param num_of_bins: Num of bins in the histogram (across all features)
    :return: A dictionary of class label -> ndarray representing the histogram for each unique class label
        The ndarray has as many dimensions as the number of features in the input data
    """
    num_of_features = len(data['feature_vectors'])
    unique_class_labels = set(data['class_labels'])
    min_max_values = [(min(f), max(f)) for f in data['feature_vectors'].values()]

    histogram_bins = {}
    histogram_bin_sizes = {}
    class_label_values = {cl: [] for cl in unique_class_labels}

    for i in range(num_of_features):
        bins, bin_size = create_bins(min_max_values[i][0], min_max_values[i][1], num_of_bins, float)
        histogram_bins[str(i)] = bins
        histogram_bin_sizes[str(i)] = bin_size

    cumulative_bin_size_multiplier = np.prod(list(histogram_bin_sizes.values()))

    for t in it.product(*histogram_bins.values()):
        value = by.bayesian_classifier(data, np.array(t))
        for cl in unique_class_labels:
            class_label_values[cl].append(value[cl] * cumulative_bin_size_multiplier)

    return {item[0]: np.array(item[1]).reshape([num_of_bins] * num_of_features)
            for item in class_label_values.items()}


def query_bin_number(data, num_of_bins, query_feature_vector, min_max_values_force=None):
    """
    This function finds the bin number (more appropriately bin vector) for the query feature vector based on the input
    data and number of bins in the histogram. Note that it is not actually required to create the histogram to perform
    this operation.

    :param data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :param num_of_bins: Num of bins in the histogram (across all features)
    :param query_feature_vector: The query vector for which we want to find bin number
    :param min_max_values_force: Force specific min, max values for each feature instead of calculating them.
        This is a list of tuples of the form [(feature1_min, feature1_max), (feature2_min, feature2_max), ...]
    :return: A list representing the 0-based bin indices in each dimension (feature)
    """
    num_of_features = len(data['feature_vectors'])
    unique_class_labels = set(data['class_labels'])
    data_size = len(data['class_labels'])

    if min_max_values_force is None:
        min_max_values = [(min(f), max(f)) for f in data['feature_vectors'].values()]
    else:
        min_max_values = min_max_values_force

    bin_number = np.zeros(num_of_features, dtype=np.int32)

    for f in range(num_of_features):
        value = query_feature_vector[f]
        min_value = min_max_values[f][0]
        max_value = min_max_values[f][1]
        bin_number[f] = assign_bin_for_value(value, min_value, max_value, num_of_bins)

    return bin_number


def query_histogram(data, num_of_bins, histogram, query_feature_vector):
    """
    This function returns the data from specific bin of each target's histogram that satisfies the given
    query feature vector.

    :param data: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list. All features have same data size
        - class_labels: A list
    :param num_of_bins: Num of bins in the histogram (across all features)
    :param histogram: A dictionary of class label -> ndarray representing the histogram for each unique class label
        The ndarray has as many dimensions as the number of features in the input data
    :param query_feature_vector: The query vector for which we want to find bin number
    :return: A dictionary from 'featureX' -> bin value for the specific bin that matches the query feature vector
        -
    """
    bin_number = query_bin_number(data, num_of_bins, query_feature_vector)
    return {h[0]: h[1][bin_number] for h in histogram.items()}
