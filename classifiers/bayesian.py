import math


def probability_density_function(value, mean, stddev):
    return 1 / (math.sqrt(2 * math.pi) * stddev) * (math.e ** (-1 / 2 * (((value - mean) / stddev) ** 2)))


def bayesian(value, mean, stddev, num_of_samples):
    return num_of_samples * probability_density_function(value, mean, stddev)
