import ast
import numpy as np
from os import listdir
import pandas as pd
import sys

"""
Usage:
build_initial_dataframe.py file_directory '[Tissue_1, Tissue_2 ...]' file_title

Directory contents: one tab separated text file per tissue, containing abundance values for all datasets. The first column name is Peptide, and the rest of the column names are the names of each dataset prefixed with the tissue (e.g. Blood_Plasma_[dataset name])

If no arguments given, defaults are used. Defaults point to the main train and test data used in these experiments.
"""

def combine_csvs(file_dir, file_paths):
    dfs = []

    for file in file_paths:
        df = pd.read_csv(file_dir + file, sep='\t', lineterminator='\r')
        dfs.append(df)

    combined_df = pd.DataFrame()
    for df in dfs:
        df.set_index('Peptide', inplace=True)
        combined_df = combined_df.join(df, how='outer')
        
    return combined_df

def filter_peptides_by_samples_and_tissues(df, min_samples, min_tissues, max_tissues, tissues, missing_val):
    df_cols = df.columns.values.tolist()
    counts = {}
    
    for tissue in tissues:
        cols = [col for col in df_cols if tissue in col] # Get corresponding list of column names
        counts[tissue] = (df[cols] != missing_val).sum(1) # count number of samples with non-imputed abundance for each protein
        
    tallys = 1 * (counts[tissues[0]] >= min_samples)
    for t in tissues[1:]:
        tallys += 1 * (counts[t] >= min_samples)

    new_df = df[(tallys >= min_tissues) & (tallys <= max_tissues)]
    return new_df

def map_tissues_to_columns(df, list_of_tissues):
    tissues_to_columns = dict([(key, []) for key in list_of_tissues])

    for column_name in df.columns.values.tolist():
        for tissue in list_of_tissues:
            if tissue in column_name:
                tissues_to_columns[tissue].append(column_name)
                continue
                
    return tissues_to_columns

### Load the data

files_dir = sys.argv[1] + '\\' if len(sys.argv) > 1 else 'F:\High_Quality_All\\'
file_paths = listdir(files_dir) 

df = combine_csvs(files_dir, file_paths)

df.dropna(axis='index', how='all', inplace=True) # drop any rows where all values are missing
if '\n' in df.index:
    df = df.drop(['\n']) # drop extra newline characters in index

### Transform and Normalize
df.iloc[:,:] = np.log2(df.iloc[:,:]) # log2 transform
df.replace([-np.inf], np.nan, inplace=True) # log2(0) returns -inf; replace with NaN to avoid skewing data

# Impute missing values as half the minimum value
df_min = df.min().min()
impute_val = df_min/2
df = df.fillna(impute_val)

tissues = ast.literal_eval(sys.argv[2]) if len(sys.argv) > 2 else['Blood_Plasma', 'Blood_Serum', 'CSF', 'Liver', 'Monocyte', 'Ovary', 'Pancreas', 'Substantia_Nigra', 'Temporal_Lobe']

tissues_to_columns = map_tissues_to_columns(df, tissues)

### Filter out peptides not observed in at least 5 samples of at least 1 tissue
df = filter_peptides_by_samples_and_tissues(df, min_samples=5, min_tissues=1, max_tissues=len(tissues), 
                                            tissues=tissues, missing_val=impute_val)

median_of_medians = df.median().median()
df /= df.median(axis=0) # divide each value by sample median
df *= median_of_medians # multiply each value by the median of medians

result_name = sys.argv[3] if len(sys.argv) > 3 else 'FullPeptideQuant.txt'
df.to_csv(result_name, sep='\t', line_terminator='\r')
