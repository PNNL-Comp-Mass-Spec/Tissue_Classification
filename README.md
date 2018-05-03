# Proteomics_Data_Processing

These are iPython notebooks that were used in a manuscript exploring the classification of human tissues via proteomics data.

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

## Peptide_Boxplots.ipynb
* Analysis of peptide variability across tissues

## Peptide_Thresholding.ipynb
* Code corresponding to 'Minimal Classifiers'
* Testing how classifiers perform on test data, with low abundance peptides removed

## build_initial_dataframe.py
* Script to create dataframes
* Ensures all dataframes go through the same cleaning and transformation steps:
 * Log2 transform
 * Impute missing values
 * Remove peptides not contained in at least 5 samples of at least 1 tissue
 * Median normalize
