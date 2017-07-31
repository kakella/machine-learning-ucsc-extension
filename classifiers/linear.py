import numpy as np


def keslerize_column(column_data):
    unique_values = set(column_data)
    unique_values_sorted = sorted(unique_values)
    num_cols = len(unique_values_sorted)

    if num_cols <= 2:
        return column_data

    keslerized_column_data = []

    for v in column_data:
        index = unique_values_sorted.index(v)
        keslerized_value = [-1] * num_cols
        keslerized_value[index] = +1
        keslerized_column_data.append(keslerized_value)

    return keslerized_column_data


def de_keslerize_columns(column_data):
    return [d.tolist().index(max(d)) for d in column_data]


def mean_square_minimizer_linear_classifier(X, T, init_value=1):
    num_rows = len(X)
    num_cols = len(X[0])
    Xa = np.column_stack(([init_value] * num_rows, X))
    pseudo_inverse_Xa = np.linalg.pinv(Xa)
    W = np.dot(pseudo_inverse_Xa, T)

    print('X: ', num_rows, num_cols)
    print('Xa: ', len(Xa), len(Xa[0]))
    print('T: ', len(T), len(T[0]))
    print('pseudo inverse: ', len(pseudo_inverse_Xa), len(pseudo_inverse_Xa[0]))
    print('W: ', len(W), len(W[0]))

    return W


def classify_using_linear_classifier(X, W, init_value=1):
    num_rows = len(X)
    Xa = np.column_stack(([init_value] * num_rows, X))
    T = np.dot(Xa, W)

    print('X: ', len(X), len(X[0]))
    print('Xa: ', len(Xa), len(Xa[0]))
    print('W: ', len(W), len(W[0]))
    print('T: ', len(T), len(T[0]))

    return T
