# Assignment 2

1. The data set that you need is in one of the sheets of the Excel file attached. The other sheets contain shaded cells meant to be filled in by you. Examine the sheets carefully and understand what must be computed or described. Except for cutting and pasting your results in the specific cells provided, do not alter the spreadsheet in any other way. This is the only recognized means of submitting this assignment.
2. Construct separate 2D histograms for male and female heights. You decide on the number of bins to use, making sure there is sufficient resolution and bin-filling. Represent height in rows, and handspan in columns -- both in ascending order of magnitude of feature. Do not use a built-in histogram program. 
3. Find the parameters of two 2D Gaussian models for the 2 PDFs to describe the data. Let the first dimension represent height, and the second dimension represent handspan. You may use built-in functions to compute these parameters, but do not use a built-in function to compute the pdf. 
4. Based on the histograms and Gaussian models, compute the likely gender (given as the probability of being female) of individuals with measurements as given below (Height in inches, handspan in centimeters). What are your observations?
Height  Handspan
69       17.5
66       22
70       21.5
69       23.5

Extra credit: Reconstruct a histogram using female model parameters that can be compared to the female histogram constructed in Part 2. Similarly, reconstruct a histogram using male model parameters.
