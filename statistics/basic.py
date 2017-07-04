import math as mt


def average(data):
    return sum(data) / len(data)


def variance(data):
    sum_of_squares = 0
    mean = average(data)
    for d in data:
        sum_of_squares += ((d - mean) ** 2)
    return sum_of_squares / (len(data) - 1)


def stddev(data):
    return mt.sqrt(variance(data))
