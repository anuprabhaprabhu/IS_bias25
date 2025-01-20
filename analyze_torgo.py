import sys, os
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

csv_pth = sys.argv[1]

severity_dict = { 'verylow': ['F03','F04','M03'],
                    'low' : ['M05','F01'],
                    'mid' : ['M01','M02','M04'],
                   'healthy' : ['FC03', 'MC03', 'MC04']}

speaker_to_class = {speaker: cls for cls, speakers in severity_dict.items() for speaker in speakers}
spkr_mos_map = {'F01': 'spkr16',  'F03':'spkr17', 'F04': 'spkr18', 'FC03':'spkr19', 'M01':'spkr20', 
                'M02': 'spkr21', 'M03':'spkr22', 'M04':'spkr23', 'M05':'spkr24', 
                'MC03':'spkr25', 'MC04':'spkr26'}

df = pd.read_csv(csv_pth)
df['spkrs'] = df['audio_path'].apply(lambda x: x.split('/')[1])
df['severity'] = df['spkrs'].map(speaker_to_class)
df['spkr_mos_map'] = df['spkrs'].map(spkr_mos_map)
df['cer'] = df['cer'].round(4)
df['wer'] = df['wer'].round(4)
print(df.head(5))
df.to_csv('torgo_final.csv', index=False)