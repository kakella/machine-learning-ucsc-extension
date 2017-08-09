from excelHelper import excel_operations as eo
from classifiers import linear as ln
from classifiers import performance as pf
from classifiers import pca
from plots import plots as plt
import numpy as np

inputExcelFile = r"Car_Data.xlsx"

training_data = eo.readExcel(inputExcelFile)[1:]

num_rows = len(training_data)

price = np.array(training_data[:, 0], dtype=str)
maintenance = np.array(training_data[:, 1], dtype=str)
doors = np.array(training_data[:, 2], dtype=str)
persons = np.array(training_data[:, 3], dtype=str)
trunk = np.array(training_data[:, 4], dtype=str)
safety = np.array(training_data[:, 5], dtype=str)
recommendation = np.array(training_data[:, 6], dtype=str)
recommendation_2_classes = np.array(['unacc' if (d == 'acc' or d == 'unacc') else 'good' for d in training_data[:, 6]],
                                    dtype=str)

price_keslerized = ln.keslerize_column(price)
maintenance_keslerized = ln.keslerize_column(maintenance)
doors_keslerized = ln.keslerize_column(doors)
persons_keslerized = ln.keslerize_column(persons)
trunk_keslerized = ln.keslerize_column(trunk)
safety_keslerized = ln.keslerize_column(safety)

X = np.column_stack((price_keslerized,
                     maintenance_keslerized,
                     doors_keslerized,
                     persons_keslerized,
                     trunk_keslerized,
                     safety_keslerized))


def run_linear_classifier(X, recommendation):
    Xa = np.column_stack(([1] * num_rows, X))
    T = ln.keslerize_column(recommendation)
    W = ln.mean_square_minimizer_linear_classifier(X, T)
    T_pred = np.dot(Xa, W)

    recommendation_prediction = ln.de_keslerize_columns(T_pred, recommendation)
    class_labels, confusion_matrix, performance_metrics = pf.evaluate_multiclass_classifier(recommendation,
                                                                                            recommendation_prediction)
    print('class_labels: \n{}\nconfusion_matrix: \n{}\nperformance_metrics: \n{}'
          .format(class_labels, confusion_matrix, performance_metrics))


mean_vector, Z, C, eigenvalues, V, Vpc, P, Punf, R, Xrec = pca.principal_component_analysis(data=X,
                                                                                            datatype='flattened')

# plt.scatter_plot(P[:, :2], recommendation, 'good', 'unacc')

X_quad = ln.generate_quadratic_X(X)
X_cubic = ln.generate_cubic_X(X)

X_quad_with_P = ln.generate_quadratic_X(P[:, :10])
X_cubic_with_P = ln.generate_cubic_X(P[:, :10])

print('\n\n*** original data, 4 class labels ***\n\n')
run_linear_classifier(X, recommendation)
print('\n\n*** original data, 2 class labels ***\n\n')
run_linear_classifier(X, recommendation_2_classes)
print('\n\n*** pca data, 4 class labels ***\n\n')
run_linear_classifier(P[:, :2], recommendation)
print('\n\n*** pca data 10 columns quadratic, 4 class labels ***\n\n')
run_linear_classifier(X_quad_with_P, recommendation)
print('\n\n*** pca data 10 columns cubic, 4 class labels ***\n\n')
run_linear_classifier(X_cubic_with_P, recommendation)
print('\n\n*** pca data 10 columns quadratic, 2 class labels ***\n\n')
run_linear_classifier(X_quad_with_P, recommendation_2_classes)
print('\n\n*** pca data 10 columns cubic, 2 class labels ***\n\n')
run_linear_classifier(X_cubic_with_P, recommendation_2_classes)
print('\n\n*** original data quadratic, 4 class labels ***\n\n')
run_linear_classifier(X_quad, recommendation)
print('\n\n*** original data cubic, 4 class labels ***\n\n')
run_linear_classifier(X_cubic, recommendation)
print('\n\n*** original data quadratic, 2 class labels ***\n\n')
run_linear_classifier(X_quad, recommendation_2_classes)
print('\n\n*** original data cubic, 2 class labels ***\n\n')
run_linear_classifier(X_cubic, recommendation_2_classes)
