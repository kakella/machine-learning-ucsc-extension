from classifiers import bayesian as by


def evaluate_classifier_result(result, true_class_label, positive_class_label, negative_class_label):
    total = sum(result.values())
    if total == 0:
        return by.indeterminate_value
    else:
        max_result = max(result.values())
        for r in result.items():
            if r[1] == max_result:
                predicted_class_label = r[0]
                if true_class_label == predicted_class_label:
                    if predicted_class_label == positive_class_label:
                        return 'TP'
                    else:
                        return 'TN'
                else:
                    if predicted_class_label == positive_class_label:
                        return 'FP'
                    else:
                        return 'FN'


def run_binary_classifiers(training_data, test_data, histogram, num_of_bins,
                           positive_class_label, negative_class_label):
    classifier_output = {
        'bayesian': {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0
        },
        'histogram': {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0
        }
    }

    for ld in test_data.items():
        true_class_label = ld[0]
        for query_feature_vector in ld[1]:

            # Run Bayesian classifier
            ldd, mean_vectors, covariances = by.prepare_data_for_bayesian_classifier(training_data)
            bayesian_values = by.bayesian_classifier(ldd, mean_vectors, covariances, query_feature_vector)
            evaluation = evaluate_classifier_result(bayesian_values, true_class_label, positive_class_label,
                                                    negative_class_label)
            if evaluation != by.indeterminate_value:
                classifier_output['bayesian'][evaluation] += 1

            # Run Histogram classifier
            histogram_values = by.histogram_classifier(training_data, num_of_bins, histogram, query_feature_vector)
            evaluation = evaluate_classifier_result(histogram_values, true_class_label, positive_class_label,
                                                    negative_class_label)
            if evaluation != by.indeterminate_value:
                classifier_output['histogram'][evaluation] += 1

    return classifier_output


def accuracy(classifier_output):
    return {
        'bayesian': (classifier_output['bayesian']['TP'] + classifier_output['bayesian']['TN']) /
                    (classifier_output['bayesian']['TP'] + classifier_output['bayesian']['TN'] +
                     classifier_output['bayesian']['FP'] + classifier_output['bayesian']['FN']),
        'histogram': (classifier_output['histogram']['TP'] + classifier_output['histogram']['TN']) /
                     (classifier_output['histogram']['TP'] + classifier_output['histogram']['TN'] +
                      classifier_output['histogram']['FP'] + classifier_output['histogram']['FN'])
    }


def sensitivity(classifier_output):
    return {
        'bayesian': classifier_output['bayesian']['TP'] /
                    (classifier_output['bayesian']['TP'] + classifier_output['bayesian']['FN']),
        'histogram': classifier_output['histogram']['TP'] /
                     (classifier_output['histogram']['TP'] + classifier_output['histogram']['FN'])
    }


def specificity(classifier_output):
    return {
        'bayesian': classifier_output['bayesian']['TN'] /
                    (classifier_output['bayesian']['FP'] + classifier_output['bayesian']['TN']),
        'histogram': classifier_output['histogram']['TN'] /
                     (classifier_output['histogram']['FP'] + classifier_output['histogram']['TN'])
    }


def positive_predictive_value(classifier_output):
    return {
        'bayesian': classifier_output['bayesian']['TP'] /
                    (classifier_output['bayesian']['FP'] + classifier_output['bayesian']['TP']),
        'histogram': classifier_output['histogram']['TP'] /
                     (classifier_output['histogram']['FP'] + classifier_output['histogram']['TP'])
    }


def negative_predictive_value(classifier_output):
    return {
        'bayesian': classifier_output['bayesian']['TN'] /
                    (classifier_output['bayesian']['FN'] + classifier_output['bayesian']['TN']),
        'histogram': classifier_output['histogram']['TN'] /
                     (classifier_output['histogram']['FN'] + classifier_output['histogram']['TN'])
    }


def evaluate_training_accuracy(training_data, test_data, histogram, num_of_bins,
                               positive_class_label, negative_class_label):
    classifier_output = run_binary_classifiers(training_data, test_data, histogram, num_of_bins,
                                               positive_class_label, negative_class_label)
    print(classifier_output)
    return accuracy(classifier_output)
