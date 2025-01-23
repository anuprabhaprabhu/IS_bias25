import sys, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

csv_pth = sys.argv[1]

df = pd.read_csv(csv_pth)

# sns.boxplot(data=df, x="severity", y="wer", hue='gender',fill=False, gap=.1, 
#             order=['healthy','very low','low','mid','high'])
# sns.barplot(data=df, x="severity", y="wer", hue='gender',fill=False, gap=.1, 
#             order=['healthy','very low','low','mid','high'])
# sns.violinplot(data=df, x="severity", y="wer")
# plt.ylim(0, 1)

####  errorbar
# grouped = df.groupby(['severity', 'spkrs'])['wer'].agg(['mean', 'std']).reset_index()
# group_labels = grouped['severity'].unique()
# plt.errorbar(x_positions, grouped['mean'], yerr=grouped['std'], fmt='o', capsize=5, color='blue', label='Mean with Std Dev')

