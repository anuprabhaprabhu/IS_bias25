import sys, os
import pandas as pd
import numpy as np


csv_pth = sys.argv[1]
df = pd.read_csv(csv_pth)

metrics = df.groupby(['severity', 'spkrs','gender'])[['wer','cer','simo','mcd_plain','mcd_dtw','mcd_dtw_sl','wer_ref','cer_ref']].agg(['mean', 'std']).reset_index()
# simo,wer,cer,mcd_plain,mcd_dtw,mcd_dtw_sl,wer_ref,cer_ref,spkrs,severity,spkr_mos_map,gender

##### to save in excel
# print(metrics)
metrics.to_excel('ua_metrics_toplot_jan23.xlsx', sheet_name='metric_all')
print(metrics)
