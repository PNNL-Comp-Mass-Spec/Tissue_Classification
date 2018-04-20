# Proteomics_Data_Processing

## TODO:
* Clean up utility functions in .py files, removing things not used for classification and moving relevant functions from MaxQuant_Postprocessing_Functions to Classification_Utils or new file called 'Plotting_Utils'

## Classification.ipynb
* Code to load and pre-process data
* Train and test various classifiers

## Classification_Utils.py
* Contains utility functions to perform basic classification processes, including cross-validation and grid searches

## MaxQuant_Postprocessing_Functions.py
* Contains functions for MaxQuant output postprocessing and graph generation
* Includes function to run file through entire pipeline

## PCA_Plotting.ipynb
* Code to produce PCA plots for train and test data, as well as a plot showing diseased samples as open circles
