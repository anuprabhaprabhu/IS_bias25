import sys, os
import pandas as pd
import numpy as np
from sklearn.utils import resample
import itertools


def calculate_statistical_parity(group1, group2):
    p_group1 = np.mean(group1)  
    p_group2 = np.mean(group2)  
    # return np.abs(p_group1 - p_group2)
    return p_group1 - p_group2


def bootstrap_stat_parity(group1, group2, n_iterations=1000):
    ci_percentile=95
    bootstrap_estimates = []
    for _ in range(n_iterations):
        group1_sample = resample(group1, n_samples= 20)
        # print(group1_sample.head(5))
        # sys.exit()
        group2_sample = resample(group2, n_samples=20)
        # print(group2_sample.head(5))
        stat_parity = calculate_statistical_parity(group1_sample['del_wer'], group2_sample['del_wer'])
        bootstrap_estimates.append(stat_parity)
    
    # Compute confidence intervals (percentile method)
    # print(np.percentile(bootstrap_estimates, [2.5, 50, 97.5]))
    lower = np.percentile(bootstrap_estimates, (100 - ci_percentile) / 2)
    upper = np.percentile(bootstrap_estimates, 100 - (100 - ci_percentile) / 2)

    original_stat_parity = calculate_statistical_parity(group1['del_wer'], group2['del_wer'])
    bias = np.mean(bootstrap_estimates) - original_stat_parity
    
    mean = np.mean(bootstrap_estimates)
    variance = np.var(bootstrap_estimates)
    
    return original_stat_parity, bias, mean, variance, lower, upper



csv_pth = sys.argv[1]
df = pd.read_csv(csv_pth)

df['del_wer'] = abs(df['wer_ref']-df['wer'])
df_wer = df[['severity', 'spkrs', 'del_wer']]
sev_gp =  ['healthy', 'verylow', 'low' ,'mid' ,'high' ]
pairs_grp = itertools.combinations(sev_gp, 2)

for x, y in pairs_grp:
    g1 = df_wer[df_wer['severity']==x].reset_index(drop=True)
    g2 = df_wer[df_wer['severity']==y].reset_index(drop=True)
    # print(g1.head())
    # print(g2.head())
    
    original_stat_parity, bias, mean, variance, lower, upper = bootstrap_stat_parity(g1, g2)
    print(x,y)
    print(f"original_stat_parity: {original_stat_parity:.4f}, lower_95percent: {lower:.4f}, upper_95percent: {upper:.4f} ")
    print(f'Bias_btw_orgiandbootstrap: {bias:.4f}, Mean_bootstrap: {mean:.4f}, Variance_bootstrap: {variance:.4f}')

    # with open(f'bootstrap_SP_ua.txt', 'a') as file:
    #     file.write(f'{x,y}\n')
    #     file.write(f"original_stat_parity: {original_stat_parity:.4f}, lower_95percent: {lower:.4f}, upper_95percent: {upper:.4f}\n ")
    #     file.write(f'Bias_btw_orgiandbootstrap: {bias:.4f}, Mean_bootstrap: {mean:.4f}, Variance_bootstrap: {variance:.4f}\n')
    #     file.write('#' * 20 + '\n')
    sys.exit()

