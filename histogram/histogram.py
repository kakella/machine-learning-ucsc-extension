import numpy as np


def create_bins(min_value, max_value, num_of_bins, datatype):
    return np.linspace(start=min_value,
                       stop=max_value,
                       num=num_of_bins,
                       dtype=datatype)


def assign_bin_for_value(value, min_value, max_value, num_of_bins):
    return int(round((num_of_bins - 1) * (value - min_value) / (max_value - min_value)))


def create_histogram(data, num_of_bins, min_value, max_value):
    bins = create_bins(min_value, max_value, num_of_bins, float)
    histogram = [0] * num_of_bins

    for d in data:
        bin_number = assign_bin_for_value(d, min_value, max_value, num_of_bins)
        histogram[bin_number] += 1

    print(histogram)
    return histogram
