from excelHelper import excel_operations as eo
from classifiers import linear as ln
from classifiers import performance as pf
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

T = ln.keslerize_column(recommendation)
T_2_classes = ln.keslerize_column(recommendation_2_classes)

X = np.column_stack((price_keslerized,
                     maintenance_keslerized,
                     doors_keslerized,
                     persons_keslerized,
                     trunk_keslerized,
                     safety_keslerized))
W = ln.mean_square_minimizer_linear_classifier(X, T)
W_2_classes = ln.mean_square_minimizer_linear_classifier(X, T_2_classes)

T_pred = np.dot(np.column_stack(([1] * num_rows, X)), W)
T_pred_2_classes = np.dot(np.column_stack(([1] * num_rows, X)), W_2_classes)

recommendation_prediction = ln.de_keslerize_columns(T_pred, recommendation)
recommendation_prediction_2_classes = ln.de_keslerize_columns(T_pred_2_classes, recommendation_2_classes)

class_labels, confusion_matrix, performance_metrics = pf.evaluate_multiclass_classifier(recommendation,
                                                                                        recommendation_prediction)
print('class_labels: \n{}\nconfusion_matrix: \n{}\nperformance_metrics: \n{}'
      .format(class_labels, confusion_matrix, performance_metrics))

class_labels, confusion_matrix, performance_metrics = pf.evaluate_multiclass_classifier(recommendation_2_classes,
                                                                                        recommendation_prediction_2_classes)
print('class_labels: \n{}\nconfusion_matrix: \n{}\nperformance_metrics: \n{}'
      .format(class_labels, confusion_matrix, performance_metrics))
