import itertools as it

import numpy as np

from classifiers import bayesian as by
from excelHelper import excelOperations as eo
from histogram import histogram as hg
from statistics import basic

num_of_bins_to_create = 32

inputExcelFile = r"Assignment_1_Data_and_Template - Original.xlsx"
outputExcelFile = r"Assignment_1_Data_and_Template - Output.xlsx"
sheets = eo.getSheetNames(inputExcelFile)
data = eo.readExcel(inputExcelFile)

X = np.array(data[:, 0] * 12 + data[:, 1], dtype=float)
T = np.array([str(g) for g in data[:, 2]])

D = [(t, x) for t, x in it.zip_longest(T, X)]
M = [x for (t, x) in D if t == 'Male']
F = [x for (t, x) in D if t == 'Female']

X50 = X[:50]
T50 = T[:50]
D50 = [(t, x) for t, x in it.zip_longest(T50, X50)]
M50 = [x for (t, x) in D50 if t == 'Male']
F50 = [x for (t, x) in D50 if t == 'Female']

sample_size_male = len(M)
sample_size_female = len(F)
sample_size_male_50 = len(M50)
sample_size_female_50 = len(F50)

print('sample_size_male, sample_size_female, sample_size_male_50, sample_size_female_50')
print(sample_size_male, sample_size_female, sample_size_male_50, sample_size_female_50)

min_height = min(X)
max_height = max(X)
# min_height_50 = min(X50)
# max_height_50 = max(X50)
min_height_50 = min_height
max_height_50 = max_height

print('min_height, max_height, min_height_50, max_height_50')
print(min_height, max_height, min_height_50, max_height_50)

histogram_male = hg.create_histogram(M, num_of_bins_to_create, min_height, max_height)
histogram_female = hg.create_histogram(F, num_of_bins_to_create, min_height, max_height)
histogram_male_50 = hg.create_histogram(M50, num_of_bins_to_create, min_height_50, max_height_50)
histogram_female_50 = hg.create_histogram(F50, num_of_bins_to_create, min_height_50, max_height_50)

mean_male = basic.average(M)
mean_female = basic.average(F)
mean_male_50 = basic.average(M50)
mean_female_50 = basic.average(F50)

print('mean_male, mean_female, mean_male_50, mean_female_50')
print(mean_male, mean_female, mean_male_50, mean_female_50)

stddev_male = basic.stddev(M)
stddev_female = basic.stddev(F)
stddev_male_50 = basic.stddev(M50)
stddev_female_50 = basic.stddev(F50)

print('stddev_male, stddev_female, stddev_male_50, stddev_female_50')
print(stddev_male, stddev_female, stddev_male_50, stddev_female_50)

#############

eo.writeExcelData(data={index: item for index, item in enumerate(histogram_female)},
                  index=[0],
                  excelFile=outputExcelFile,
                  sheetName='Classifiers - Full Data',
                  startRow=5,
                  startCol=3)

eo.writeExcelData(data={index: item for index, item in enumerate(histogram_male)},
                  index=[0],
                  excelFile=outputExcelFile,
                  sheetName='Classifiers - Full Data',
                  startRow=6,
                  startCol=3)

eo.writeExcelData(data={index: item for index, item in enumerate(histogram_female_50)},
                  index=[0],
                  excelFile=outputExcelFile,
                  sheetName='Classifiers - Partial Data',
                  startRow=5,
                  startCol=3)

eo.writeExcelData(data={index: item for index, item in enumerate(histogram_male_50)},
                  index=[0],
                  excelFile=outputExcelFile,
                  sheetName='Classifiers - Partial Data',
                  startRow=6,
                  startCol=3)

#############

queries = eo.readExcel(excelFile=inputExcelFile,
                       sheetName='Classifiers - Full Data',
                       startRow=17,
                       endRow=17,
                       startCol=2,
                       endCol=7).astype(float)

queries_50 = eo.readExcel(excelFile=inputExcelFile,
                          sheetName='Classifiers - Partial Data',
                          startRow=17,
                          endRow=17,
                          startCol=2,
                          endCol=7).astype(float)

histogram_result = []
bayesian_result = []
histogram_result_50 = []
bayesian_result_50 = []

for q in queries:
    bin_number = hg.assign_bin_for_value(q, min_height, max_height, num_of_bins_to_create)
    if (histogram_female[bin_number] + histogram_male[bin_number]) == 0:
        histogram_result.append(0)
    else:
        histogram_result.append(
            histogram_female[bin_number] / (histogram_female[bin_number] + histogram_male[bin_number]))

    bayesian_male = by.bayesian(q, mean_male, stddev_male, sample_size_male)
    bayesian_female = by.bayesian(q, mean_female, stddev_female, sample_size_female)
    if (bayesian_male + bayesian_female) == 0:
        bayesian_result.append(0)
    else:
        bayesian_result.append(bayesian_female / (bayesian_male + bayesian_female))

for q in queries_50:
    bin_number = hg.assign_bin_for_value(q, min_height_50, max_height_50, num_of_bins_to_create)
    if (histogram_female_50[bin_number] + histogram_male_50[bin_number]) == 0:
        histogram_result_50.append(0)
    else:
        histogram_result_50.append(
            histogram_female_50[bin_number] / (histogram_female_50[bin_number] + histogram_male_50[bin_number]))

    bayesian_male = by.bayesian(q, mean_male_50, stddev_male_50, sample_size_male_50)
    bayesian_female = by.bayesian(q, mean_female_50, stddev_female_50, sample_size_female_50)
    if (bayesian_male + bayesian_female) == 0:
        bayesian_result_50.append(0)
    else:
        bayesian_result_50.append(bayesian_female / (bayesian_male + bayesian_female))

print('histogram_result')
print(histogram_result)

print('histogram_result_50')
print(histogram_result_50)

print('bayesian_result')
print(bayesian_result)

print('bayesian_result_50')
print(bayesian_result_50)
