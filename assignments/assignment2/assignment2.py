from classifiers import bayesian as by
from excelHelper import excel_operations as eo
from histogram import binner as bn
from histogram import histogram as hg
from prepare_data import classification_problems as cp

inputExcelFile = r"Assignment_2_Data_and_Template - Input.xlsx"
outputExcelFile = r"Assignment_2_Data_and_Template - Output - Alternate.xlsx"

raw_data = eo.readExcel(inputExcelFile)
label_column_index = 0
feature_count = raw_data[0].size - 1

data = cp.get_data(raw_data, label_column_index, feature_count)
num_of_bins = bn.calculate_number_of_bins_nd(data['feature_vectors'])

# Based on experimentation, although the above code returns num_of_bins = 10,
# I found 8 to be more reliable for the given data set
num_of_bins = 8

histogram = hg.create_histogram(data, num_of_bins)

print('histogram')
print(histogram)

print('histogram classifier output')
print('[69, 17.5]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [69, 17.5], 'Female'))
print('[66, 22]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [66, 22], 'Female'))
print('[70, 21.5]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [70, 21.5], 'Female'))
print('[69, 23.5]', by.histogram_classifier_for_target(data, num_of_bins, histogram, [69, 23.5], 'Female'))

print('bayesian classifier output')
print('[69, 17.5]', by.bayesian_classifier_for_target(data, [69, 17.5], 'Female'))
print('[66, 22]', by.bayesian_classifier_for_target(data, [66, 22], 'Female'))
print('[70, 21.5]', by.bayesian_classifier_for_target(data, [70, 21.5], 'Female'))
print('[69, 23.5]', by.bayesian_classifier_for_target(data, [69, 23.5], 'Female'))

bayesian_histogram = hg.create_histogram_from_bayesian_params(data, num_of_bins)

print('bayesian histogram')
print(bayesian_histogram)

print('checking the bayesian histogram classifier output against bayesian classifier output')
print('[69, 17.5]', by.histogram_classifier_for_target(data, num_of_bins, bayesian_histogram, [69, 17.5], 'Female'))
print('[66, 22]', by.histogram_classifier_for_target(data, num_of_bins, bayesian_histogram, [66, 22], 'Female'))
print('[70, 21.5]', by.histogram_classifier_for_target(data, num_of_bins, bayesian_histogram, [70, 21.5], 'Female'))
print('[69, 23.5]', by.histogram_classifier_for_target(data, num_of_bins, bayesian_histogram, [69, 23.5], 'Female'))

eo.writeExcelData(data={index: item for index, item in enumerate(histogram['Female'])},
                  excelFile=outputExcelFile,
                  sheetName='Female Histogram',
                  startRow=7,
                  startCol=2)
eo.writeExcelData(data={index: item for index, item in enumerate(histogram['Male'])},
                  excelFile=outputExcelFile,
                  sheetName='Male Histogram',
                  startRow=7,
                  startCol=2)

eo.writeExcelData(data={index: item for index, item in enumerate(bayesian_histogram['Female'])},
                  excelFile=outputExcelFile,
                  sheetName='Reconstructed Female Histogram',
                  startRow=7,
                  startCol=2)
eo.writeExcelData(data={index: item for index, item in enumerate(bayesian_histogram['Male'])},
                  excelFile=outputExcelFile,
                  sheetName='Reconstructed Male Histogram',
                  startRow=7,
                  startCol=2)
