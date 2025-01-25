import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

csv_pth = sys.argv[1]
df = pd.read_csv(csv_pth)
df[['wer','cer','wer_ref','cer_ref']] = df[['wer','cer','wer_ref','cer_ref']]*100
# df = df.round(2)
#########  CER
spkr_wise = df.groupby(['severity','spkrs'])[['cer','cer_ref']].mean().reset_index()


healthy = spkr_wise[spkr_wise['severity']=='healthy']
very_low = spkr_wise[spkr_wise['severity']=='very low']
low = spkr_wise[spkr_wise['severity']=='low']
mid = spkr_wise[spkr_wise['severity']=='mid']
high = spkr_wise[spkr_wise['severity']=='high']

h = plt.scatter( healthy['cer'],healthy['cer_ref'], s=100, c='m', marker='.')
m = plt.scatter( very_low['cer'],very_low['cer_ref'],s=100, c='g', marker='.')
l = plt.scatter( low['cer'],low['cer_ref'], s=100, c='b', marker='.')
vl = plt.scatter( mid['cer'],mid['cer_ref'],s=100, c='r', marker='.')
hi = plt.scatter( high['cer'],high['cer_ref'],s=100, c='k', marker='.')

###########  smiple slope plot
# plt.plot([min(y), max(y)], [df['avg_proba'].min(), df['avg_proba'].max()], 'k-')


###########  kind of linear regression plot
slope, intercept, r_value, p_value, std_err = stats.linregress(spkr_wise['cer'],spkr_wise['cer_ref'])
# x_fit = np.linspace(min(spkr_wise['wer']), max(spkr_wise['wer_ref']), 100)
x_fit = np.linspace(min(spkr_wise['cer_ref']), max(spkr_wise['cer']), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, color='c', linestyle='-', linewidth=2, label='Fitted Line')
 #######################        

plt.title('Scatter plot of CER')
plt.xlabel('CER of generated samples')
plt.ylabel('CER of refernce samples')
plt.xticks(np.arange(0, 120, 20))
plt.yticks(np.arange(0, 120, 20))
plt.legend((h,m,l,vl,hi),('healthy','very low','low','mid','high'))
plt.show()


#################   WER

spkr_wise = df.groupby(['severity','spkrs'])[['wer','wer_ref']].mean().reset_index()


healthy = spkr_wise[spkr_wise['severity']=='healthy']
very_low = spkr_wise[spkr_wise['severity']=='very low']
low = spkr_wise[spkr_wise['severity']=='low']
mid = spkr_wise[spkr_wise['severity']=='mid']
high = spkr_wise[spkr_wise['severity']=='high']

h = plt.scatter( healthy['wer'],healthy['wer_ref'], s=100, c='m', marker='.')
m = plt.scatter( very_low['wer'],very_low['wer_ref'],s=100, c='g', marker='.')
l = plt.scatter( low['wer'],low['wer_ref'], s=100, c='b', marker='.')
vl = plt.scatter( mid['wer'],mid['wer_ref'],s=100, c='r', marker='.')
hi = plt.scatter( high['wer'],high['wer_ref'],s=100, c='k', marker='.')

###########  smiple slope plot
# plt.plot([min(y), max(y)], [df['avg_proba'].min(), df['avg_proba'].max()], 'k-')


###########  kind of linear regression plot
slope, intercept, r_value, p_value, std_err = stats.linregress(spkr_wise['wer'],spkr_wise['wer_ref'])
# x_fit = np.linspace(min(spkr_wise['wer']), max(spkr_wise['wer_ref']), 100)
x_fit = np.linspace(min(spkr_wise['wer_ref']), max(spkr_wise['wer']), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, color='c', linestyle='-', linewidth=2, label='Fitted Line')
 #######################        

plt.title('Scatter plot of WER')
plt.xlabel('WER of generated samples')
plt.ylabel('WER of refernce samples')
plt.xticks(np.arange(0, 120, 20))
plt.yticks(np.arange(0, 140, 20))
plt.legend((h,m,l,vl,hi),('healthy','very low','low','mid','high'))
plt.show()