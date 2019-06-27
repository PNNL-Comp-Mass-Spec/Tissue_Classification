# Tissue_Classification

This is the repository for computational methods associated with the paper "Individual Variability of Protein Expression in Human Tissues" which can be found at https://www.ncbi.nlm.nih.gov/pubmed/30300549.

These are iPython notebooks and associated scripts that were used in a manuscript exploring the classification of human tissues via proteomics data. There are a variety of notebooks which are used to make figures for the manuscript. Each starts from the same basic dataframe. Therefore, the repository should be fairly self-contained, with this df as a starting point. See requirements.txt for required software packages and versions.


## Notebooks - used to make figures in the manuscript
Please upzip FullPeptideQuant.txt.zip in your local repository to reproduce all the results.

### Model_Finalization notebooks
* Feature selection and classifier parameter tuning
* Trained_Models directory contains the compressed finalized models, which can be loaded directly into a notebook. This way the finalized models can be used without running the grid searches, which are time-intensive.

### Classification.ipynb
* Code to load and pre-process data
* Train and test various classifiers. The goal of these classifiers is to see whether we can correctly predict the source tissue of a proteomics sample.

### Cohort_Sizes.ipynb
* Classifying blood plasma and serum with increasingly larger training sets

### tSNE_PCA.ipynb
* Code to produce tSNE and PCA plots for train and test data
* Includes plots showing diseased samples as open circles, and plots with cell line datasets

### Peptide_Boxplots.ipynb
* Analysis of peptide variability across tissues. Used to create a figure displaying the distinct expression patterns of four archetypal peptides.

### Peptide_Thresholding.ipynb
* Code corresponding to 'Minimal Classifiers'
* Testing how classifiers perform on test data, with low abundance peptides removed


## Auxiliary Scripts
### Classification_Utils.py
* Contains utility functions to perform basic classification processes, including cross-validation and grid searches

### build_initial_dataframe.py
* Script to create dataframes
* Ensures all dataframes go through the same cleaning and transformation steps:
 * Log2 transform
 * Impute missing values
 * Remove peptides not contained in at least 5 samples of at least 1 tissue
 * Median normalize
