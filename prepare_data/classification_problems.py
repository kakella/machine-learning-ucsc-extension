import numpy as np


def get_data(raw_data, label_column_index, feature_count):
    """
    This function gets raw data in the form of a 2d array and separates it into lists of the following:
    - Each feature from the feature vector
    - Class label

    :param raw_data: Raw data in 2d form
    :param label_column_index: Index of the column representing class label (should be first column)
    :param feature_count: Number of features to read from the data
    :return: A dictionary containing the following:
        - feature_vectors: A dictionary from 'featureX' -> feature list
        - class_labels: A list
    """
    data = np.array(raw_data)
    class_labels = data[:, label_column_index]
    feature_vectors = {}

    for i in range(feature_count):
        feature_vectors['feature' + str(i)] = np.array(data[:, i + 1], dtype=float)

    return {
        'feature_vectors': feature_vectors,
        'class_labels': class_labels
    }
