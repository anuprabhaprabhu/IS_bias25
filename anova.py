import sys, os
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

csv_pth = sys.argv[1]
df = pd.read_csv(csv_pth)
# df[['wer','cer','wer_ref','cer_ref']] = df[['wer','cer','wer_ref','cer_ref']]*100
# df = df.round(2)

grouped = [df[df['severity'] == c]['wer'] for c in df['severity'].unique()]
f_stat, p_value = stats.f_oneway(*grouped)

print(f'WER')
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

grouped = [df[df['severity'] == c]['cer'] for c in df['severity'].unique()]
f_stat, p_value = stats.f_oneway(*grouped)

print(f'CER')
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

grouped = [df[df['severity'] == c]['simo'] for c in df['severity'].unique()]
f_stat, p_value = stats.f_oneway(*grouped)

print(f'SIM-o')
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")

# grouped = [df[df['severity'] == c][['wer','cer','simo']] for c in df['severity'].unique()]

# for column in grouped.select_dtypes(include='number').columns:
#     print(f"ANOVA for {column}:")
#     # Fit the model using OLS (including all factors and their interactions)
#     formula = f"{column} ~ C(Factor_A) + C(Factor_B) + C(Factor_C) + C(Factor_A):C(Factor_B) + C(Factor_A):C(Factor_C) + C(Factor_B):C(Factor_C) + C(Factor_A):C(Factor_B):C(Factor_C)"
#     model = ols(formula, data=grouped).fit()
#     anova_results = anova_lm(model)
#     print(anova_results)
#     print("\n")