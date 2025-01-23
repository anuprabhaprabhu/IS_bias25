import sys, os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import root_mean_squared_error

csv_pth = sys.argv[1]
df = pd.read_csv(csv_pth)

dysar_dict = { 'Spastic': ['F02','F03','F05','M01','M05','M07','M14','M16'],
                    'Mixed' : ['M10','M12'],
                    'Athetoid' : ['F04','M11'],
                   'healthy' : ['CF02', 'CM05', 'CM08']}

dysar_dict_cls = {speaker: cls for cls, speakers in dysar_dict.items() for speaker in speakers}     
df['dysar_type'] = df['spkrs'].map(dysar_dict_cls)

##############  wer all speakers 
# df[['wer','cer','wer_ref','cer_ref']] = df[['wer','cer','wer_ref','cer_ref']]*100
spkr_wise = df.groupby(['spkrs'])[['wer','cer','wer_ref','cer_ref']].mean().reset_index()
severity_wise = df.groupby(['severity'])[['wer','cer','wer_ref','cer_ref']].mean().reset_index()
# simo,wer,cer,mcd_plain,mcd_dtw,mcd_dtw_sl,wer_ref,cer_ref,spkrs,severity,spkr_mos_map,gender

print(spkr_wise)
print(severity_wise)

pc = pearsonr(spkr_wise['wer'], spkr_wise['wer_ref'] )
sp = spearmanr(spkr_wise['wer'], spkr_wise['wer_ref'] )
tau, p_value = kendalltau(spkr_wise['wer'], spkr_wise['wer_ref'] )
rmse = root_mean_squared_error(spkr_wise['wer'], spkr_wise['wer_ref'])
print(f'Speaker wise - wer ')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

pc = pearsonr(severity_wise['wer'], severity_wise['wer_ref'] )
sp = spearmanr(severity_wise['wer'], severity_wise['wer_ref'] )
tau, p_value = kendalltau(severity_wise['wer'], severity_wise['wer_ref'] )
rmse = root_mean_squared_error(severity_wise['wer'], severity_wise['wer_ref'])
print(f'Severity wise - wer')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

###################  cer all speakers 
pc = pearsonr(spkr_wise['cer'], spkr_wise['cer_ref'] )
sp = spearmanr(spkr_wise['cer'], spkr_wise['cer_ref'] )
tau, p_value = kendalltau(spkr_wise['cer'], spkr_wise['cer_ref'] )
rmse = root_mean_squared_error(spkr_wise['cer'], spkr_wise['cer_ref'])
print(f'Speaker wise - cer ')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

pc = pearsonr(severity_wise['cer'], severity_wise['cer_ref'] )
sp = spearmanr(severity_wise['cer'], severity_wise['cer_ref'] )
tau, p_value = kendalltau(severity_wise['cer'], severity_wise['cer_ref'] )
rmse = root_mean_squared_error(severity_wise['cer'], severity_wise['cer_ref'])
print(f'Severity wise - cer')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)


################    wer - male 
df_male = df[df['gender']=='male']
spkr_wise = df_male.groupby(['spkrs'])[['wer','cer','wer_ref','cer_ref']].mean().reset_index()
severity_wise = df_male.groupby(['severity'])[['wer','cer','wer_ref','cer_ref']].mean().reset_index()
# simo,wer,cer,mcd_plain,mcd_dtw,mcd_dtw_sl,wer_ref,cer_ref,spkrs,severity,spkr_mos_map,gender

print(spkr_wise)
print(severity_wise)

pc = pearsonr(spkr_wise['wer'], spkr_wise['wer_ref'] )
sp = spearmanr(spkr_wise['wer'], spkr_wise['wer_ref'] )
tau, p_value = kendalltau(spkr_wise['wer'], spkr_wise['wer_ref'] )
rmse = root_mean_squared_error(spkr_wise['wer'], spkr_wise['wer_ref'])
print(f'Speaker - Male - wer ')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

pc = pearsonr(severity_wise['wer'], severity_wise['wer_ref'] )
sp = spearmanr(severity_wise['wer'], severity_wise['wer_ref'] )
tau, p_value = kendalltau(severity_wise['wer'], severity_wise['wer_ref'] )
rmse = root_mean_squared_error(severity_wise['wer'], severity_wise['wer_ref'])
print(f'Severity - Male- wer')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

###################  cer -- male
pc = pearsonr(spkr_wise['cer'], spkr_wise['cer_ref'] )
sp = spearmanr(spkr_wise['cer'], spkr_wise['cer_ref'] )
tau, p_value = kendalltau(spkr_wise['cer'], spkr_wise['cer_ref'] )
rmse = root_mean_squared_error(spkr_wise['cer'], spkr_wise['cer_ref'])
print(f'Speaker male - cer ')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

pc = pearsonr(severity_wise['cer'], severity_wise['cer_ref'] )
sp = spearmanr(severity_wise['cer'], severity_wise['cer_ref'] )
tau, p_value = kendalltau(severity_wise['cer'], severity_wise['cer_ref'] )
rmse = root_mean_squared_error(severity_wise['cer'], severity_wise['cer_ref'])
print(f'Severity male - cer')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)


################    wer - female 
df_male = df[df['gender']=='female']
spkr_wise = df_male.groupby(['spkrs'])[['wer','cer','wer_ref','cer_ref']].mean().reset_index()
severity_wise = df_male.groupby(['severity'])[['wer','cer','wer_ref','cer_ref']].mean().reset_index()
# simo,wer,cer,mcd_plain,mcd_dtw,mcd_dtw_sl,wer_ref,cer_ref,spkrs,severity,spkr_mos_map,gender

print(spkr_wise)
print(severity_wise)

pc = pearsonr(spkr_wise['wer'], spkr_wise['wer_ref'] )
sp = spearmanr(spkr_wise['wer'], spkr_wise['wer_ref'] )
tau, p_value = kendalltau(spkr_wise['wer'], spkr_wise['wer_ref'] )
rmse = root_mean_squared_error(spkr_wise['wer'], spkr_wise['wer_ref'])
print(f'Speaker - female - wer ')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

pc = pearsonr(severity_wise['wer'], severity_wise['wer_ref'] )
sp = spearmanr(severity_wise['wer'], severity_wise['wer_ref'] )
tau, p_value = kendalltau(severity_wise['wer'], severity_wise['wer_ref'] )
rmse = root_mean_squared_error(severity_wise['wer'], severity_wise['wer_ref'])
print(f'Severity - female - wer')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

###################  cer -- female
pc = pearsonr(spkr_wise['cer'], spkr_wise['cer_ref'] )
sp = spearmanr(spkr_wise['cer'], spkr_wise['cer_ref'] )
tau, p_value = kendalltau(spkr_wise['cer'], spkr_wise['cer_ref'] )
rmse = root_mean_squared_error(spkr_wise['cer'], spkr_wise['cer_ref'])
print(f'Speaker female - cer ')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

pc = pearsonr(severity_wise['cer'], severity_wise['cer_ref'] )
sp = spearmanr(severity_wise['cer'], severity_wise['cer_ref'] )
tau, p_value = kendalltau(severity_wise['cer'], severity_wise['cer_ref'] )
rmse = root_mean_squared_error(severity_wise['cer'], severity_wise['cer_ref'])
print(f'Severity male - cer')
print(pc)
print(sp)
print(tau, p_value)
print(rmse)

print("Completted !!!!!!")