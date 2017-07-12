import numpy as np
import scipy.stats as ss


def calculate_bin_size_1d(sample):
    """
    This function calculates the optimal bin size for creating a 1D histogram.
    The following algorithm is applied:
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule

    :param sample: The sample data for a single feature
    :return: The optimal size of a bin
    """
    return 2 * ss.iqr(sample) * (len(sample) ** (-1 / 3))


def calculate_number_of_bins_1d(sample):
    """
    This function calculates the optimal number of bins for creating a 1D histogram

    :param sample: The sample data for a single feature
    :return: The optimal number of bins
    """
    return (max(sample) - min(sample)) / calculate_bin_size_1d(sample)


def calculate_number_of_bins_nd(feature_vectors):
    """
    This function calculates the optimal number of bins for creating a multi-dimensional histogram
    It does this by calculating the optimal number of bins for each feature independently, and then averaging them out
    This is done because we want to have the same number of bins for every feature
    (to keep the Bayesian formulas for comparison with histogram simple)

    :param feature_vectors: The sample data for all features
    :return: The optimal number of bins
    """
    return int(round(np.mean([calculate_number_of_bins_1d(f) for f in feature_vectors.values()])))
