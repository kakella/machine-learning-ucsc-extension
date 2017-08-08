import numpy as np


def keslerize_column(column_data):
    unique_values = set(column_data)
    unique_values_sorted = sorted(unique_values)
    num_cols = len(unique_values_sorted)

    # if num_cols <= 2:
    #     return column_data

    # print('keslerizing into {} columns'.format(num_cols))

    keslerized_column_data = []

    for v in column_data:
        index = unique_values_sorted.index(v)
        keslerized_value = [-1] * num_cols
        keslerized_value[index] = +1
        keslerized_column_data.append(keslerized_value)

    return np.array(keslerized_column_data)


def de_keslerize_columns(keslerized_output_data, original_input_data=None):
    indexed_list = [d.tolist().index(max(d)) for d in keslerized_output_data]
    if original_input_data is None:
        return indexed_list
    else:
        unique_values_sorted = sorted(set(original_input_data))
        return [unique_values_sorted[d] for d in indexed_list]


def mean_square_minimizer_linear_classifier(X, T, init_value=1):
    num_rows = len(X)
    num_cols = len(X[0])
    Xa = np.column_stack(([init_value] * num_rows, X))
    pseudo_inverse_Xa = np.linalg.pinv(Xa)
    W = np.dot(pseudo_inverse_Xa, T)

    print('X: ', X.shape)
    print('Xa: ', Xa.shape)
    print('T: ', T.shape)
    print('pseudo inverse: ', pseudo_inverse_Xa.shape)
    print('W: ', W.shape)

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
