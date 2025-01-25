import sys, os
import pandas as pd
import numpy as np
import scipy.stats as stats


csv_pth = sys.argv[1]
df = pd.read_csv(csv_pth)
# df[['wer','cer','wer_ref','cer_ref']] = df[['wer','cer','wer_ref','cer_ref']]*100
# df = df.round(2)

grouped = [df[df['severity'] == c]['wer'] for c in df['severity'].unique()]
f_stat, p_value = stats.kruskal(*grouped)

print(f'WER')
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

grouped = [df[df['severity'] == c]['cer'] for c in df['severity'].unique()]
f_stat, p_value = stats.kruskal(*grouped)

print(f'CER')
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

grouped = [df[df['severity'] == c]['simo'] for c in df['severity'].unique()]
f_stat, p_value = stats.kruskal(*grouped)

print(f'SIM-o')
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")