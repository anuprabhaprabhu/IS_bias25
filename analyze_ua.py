import sys, os
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt

csv_pth = sys.argv[1]

severity_dict = { 'verylow': ['F05','M08','M09','M10','M14'],
                    'low' : ['M05','F04','M11'],
                    'mid' : ['F02','M07','M16'],
                   'high' : ['F03','M01','M04','M12'],
                   'healthy' : ['CF02', 'CM05', 'CM08']}

speaker_to_class = {speaker: cls for cls, speakers in severity_dict.items() for speaker in speakers}
spkr_mos_map = {'CF02': 'spkr1','CM05': 'spkr2', 'CM08': 'spkr3', 'F02': 'spkr4', 'F03': 'spkr5', 
                'F04': 'spkr6',  'F05':'spkr7', 'M01': 'spkr8', 'M05':'spkr9', 'M07':'spkr10', 
                'M10': 'spkr11', 'M11':'spkr12', 'M12':'spkr13', 'M14':'spkr14', 'M16':'spkr15'}

df = pd.read_csv(csv_pth)
df['spkrs'] = df['audio_path'].str.extract(r'([^/]+)_\d+\.wav')
df['severity'] = df['spkrs'].map(speaker_to_class)
df['spkr_mos_map'] = df['spkrs'].map(spkr_mos_map)
df['cer'] = df['cer'].round(4)
df['wer'] = df['wer'].round(4)
df['mcd_plain'] = df['mcd_plain'].round(4)
df['mcd_dtw'] = df['mcd_dtw'].round(4)
df['mcd_dtw_sl'] = df['mcd_dtw_sl'].round(4)
df['cer_ref'] = df['cer_ref'].round(4)
df['wer_ref'] = df['wer_ref'].round(4)

df['gender'] = df['audio_path'].apply(lambda x: 'female' if 'F' in x.split('/')[-1] else 'male')
print(df.head(5))
# sys.exit()
df.to_csv('ua_final_jan23.csv', index=False)