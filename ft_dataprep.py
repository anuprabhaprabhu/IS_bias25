import os, sys
import numpy as np
import pandas as pd 


csv_pth = sys.argv[1]

df = pd.read_csv(csv_pth)
unispkrs = df['audio_path'].apply(lambda x: x.split('/')[1]).unique()
print(unispkrs)

unispkrs = ['M04','M01','F04' ,'M03','M05', 'FC03', 'MC03']
with open('gen_text_5.txt', 'r') as file:
    new_column_values = file.readlines()
new_column_values = [line.strip() for line in new_column_values]

df = df.drop('gen_text', axis=1)
df = df.drop('out_path', axis=1)
df_final = pd.DataFrame()
for spk in unispkrs:
    print( spk)
    df_spkr = df[df['audio_path'].apply(lambda x: x.split('/')[1])==spk]
    df_spkr['gen_text'] = new_column_values
    df_spkr['out_path'] = [f"{spk}_{i}_gen.wav" for i in range(21, 41)]
    df_final = pd.concat([df_final, df_spkr], ignore_index=True)
    # print(df_spkr)
    # sys.exit()
df_final.to_csv('torgo_gen5.csv', sep=',', index=False)
print('Completted !!!!')
    
