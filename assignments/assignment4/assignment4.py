from excelHelper import excel_operations as eo
from classifiers import linear as ln
from classifiers import performance as pf
import numpy as np

inputExcelFile = r"Assignment_4_Data_and_Template - Input.xlsx"
outputExcelFile = r"Assignment_4_Data_and_Template - Output.xlsx"

training_data = eo.readExcel(inputExcelFile)
num_cols = len(training_data[0])
num_rows = len(training_data)

failure_column = training_data[:, num_cols - 2]
type_column = training_data[:, num_cols - 1]
type_keslerized = ln.keslerize_column(type_column)

T = np.column_stack((failure_column, type_keslerized))
X = training_data[:, :num_cols - 2]
W = ln.mean_square_minimizer_linear_classifier(X, T)

W_failure = W[:, 0]
W_type = W[:, 1:]

W_failure_for_excel = []
for w in W_failure:
    W_failure_for_excel.append([w])

test_data = eo.readExcel(inputExcelFile,
                         sheetName="To be classified",
                         startRow=5,
                         endRow=54,
                         startCol=1,
                         endCol=15)
test_results = ln.classify_using_linear_classifier(test_data, W)
T_failure = [1 if r > 0 else -1 for r in test_results[:, 0]]
T_type = ln.de_keslerize_columns(test_results[:, 1:])
T_classified = np.column_stack((T_failure, T_type))

training_results = ln.classify_using_linear_classifier(X, W)
TR_failure = [1 if r > 0 else -1 for r in training_results[:, 0]]
TR_type = ln.de_keslerize_columns(training_results[:, 1:])

failure_result = pf.evaluate_binary_classifier(failure_column, TR_failure, 1, -1)
failure_metrics = pf.binary_classifier_performance_metrics(failure_result)
print('failure classification')
print(failure_result)
print(failure_metrics)

type_labels, type_result = pf.evaluate_multiclass_classifier(type_column, TR_type)
type_metrics = pf.multiclass_classifier_performance_metrics(type_result)
print('type classification')
print(type_labels, type_result)
print(type_metrics)


# eo.writeExcelData(data=W_failure_for_excel,
#                   excelFile=outputExcelFile,
#                   sheetName='Classifiers',
#                   startRow=5,
#                   startCol=1)

# eo.writeExcelData(data=W_type,
#                   excelFile=outputExcelFile,
#                   sheetName='Classifiers',
#                   startRow=5,
#                   startCol=5)

# eo.writeExcelData(data=T_classified,
#                   excelFile=outputExcelFile,
#                   sheetName='To be classified',
#                   startRow=5,
#                   startCol=16)

# eo.writeExcelData(data=type_result,
#                   excelFile=outputExcelFile,
#                   sheetName='Performance',
#                   startRow=19,
#                   startCol=3)
