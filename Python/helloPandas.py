import pandas as pd
frame = pd.read_csv('..\Data\data_sample_example.tsv', sep='\t', header=0)
print frame
print frame.shape
