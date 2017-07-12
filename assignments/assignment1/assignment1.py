from classifiers import bayesian as by
from excelHelper import excel_operations as eo
from histogram import histogram as hg
from prepare_data import classification_problems as cp

num_of_bins_to_create = 32

inputExcelFile = r"Assignment_1_Data_and_Template - Input.xlsx"
outputExcelFile = r"Assignment_1_Data_and_Template - Output.xlsx"

raw_data = eo.readExcel(inputExcelFile)
raw_data_cleaned = [[row[2], row[0] * 12 + row[1]] for row in raw_data]
label_column_index = 0
feature_count = 1

data = cp.get_data(raw_data_cleaned, label_column_index, feature_count)
data50 = cp.get_data(raw_data_cleaned[:50], label_column_index, feature_count)
num_of_bins = 32

min_height = min([d[1] for d in raw_data_cleaned])
max_height = max([d[1] for d in raw_data_cleaned])

print('min, max: ', min_height, max_height)

histogram = hg.create_histogram(data, num_of_bins)
histogram50 = hg.create_histogram(data50, num_of_bins, [(min_height, max_height)])

print('histograms')
print(histogram)
print(histogram50)

print('histogram classifier output - full data')
print('[55]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [55], 'Female'))
print('[60]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [60], 'Female'))
print('[65]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [65], 'Female'))
print('[70]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [70], 'Female'))
print('[75]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [75], 'Female'))
print('[80]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [80], 'Female'))

print('bayesian classifier output - full data')
print('[55]', by.bayesian_classifier_for_target(data, [55], 'Female'))
print('[60]', by.bayesian_classifier_for_target(data, [60], 'Female'))
print('[65]', by.bayesian_classifier_for_target(data, [65], 'Female'))
print('[70]', by.bayesian_classifier_for_target(data, [70], 'Female'))
print('[75]', by.bayesian_classifier_for_target(data, [75], 'Female'))
print('[80]', by.bayesian_classifier_for_target(data, [80], 'Female'))

print('histogram classifier output - partial data')
print('[55]', by.histogram_classifier_for_target(data50, num_of_bins, histogram50, [55], 'Female',
                                                 [(min_height, max_height)]))
print('[60]', by.histogram_classifier_for_target(data50, num_of_bins, histogram50, [60], 'Female',
                                                 [(min_height, max_height)]))
print('[65]', by.histogram_classifier_for_target(data50, num_of_bins, histogram50, [65], 'Female',
                                                 [(min_height, max_height)]))
print('[70]', by.histogram_classifier_for_target(data50, num_of_bins, histogram50, [70], 'Female',
                                                 [(min_height, max_height)]))
print('[75]', by.histogram_classifier_for_target(data50, num_of_bins, histogram50, [75], 'Female',
                                                 [(min_height, max_height)]))
print('[80]', by.histogram_classifier_for_target(data50, num_of_bins, histogram50, [80], 'Female',
                                                 [(min_height, max_height)]))

print('bayesian classifier output - partial data')
print('[55]', by.bayesian_classifier_for_target(data50, [55], 'Female'))
print('[60]', by.bayesian_classifier_for_target(data50, [60], 'Female'))
print('[65]', by.bayesian_classifier_for_target(data50, [65], 'Female'))
print('[70]', by.bayesian_classifier_for_target(data50, [70], 'Female'))
print('[75]', by.bayesian_classifier_for_target(data50, [75], 'Female'))
print('[80]', by.bayesian_classifier_for_target(data50, [80], 'Female'))

bayesian_histogram = hg.create_histogram_from_bayesian_params(data, num_of_bins)

print('bayesian histogram')
print(bayesian_histogram)

# eo.writeExcelData(data=[{index: item for index, item in enumerate(histogram['Female'])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Classifiers - Full Data',
#                   startRow=5,
#                   startCol=3)
#
# eo.writeExcelData(data=[{index: item for index, item in enumerate(histogram['Male'])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Classifiers - Full Data',
#                   startRow=6,
#                   startCol=3)
#
# eo.writeExcelData(data=[{index: item for index, item in enumerate(histogram50['Female'])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Classifiers - Partial Data',
#                   startRow=5,
#                   startCol=3)
#
# eo.writeExcelData(data=[{index: item for index, item in enumerate(histogram50['Male'])}],
#                   excelFile=outputExcelFile,
#                   sheetName='Classifiers - Partial Data',
#                   startRow=6,
#                   startCol=3)
