import os, sys 
import pandas as pd 

orgi_ctrl = sys.argv[1]  # original
orgi_dysar = sys.argv[2]  # original
new = sys.argv[3]  # new

df1 = pd.read_csv(orgi_ctrl)
df2 = pd.read_csv(orgi_dysar)
df3 = pd.read_csv(new)

print(df3.shape)
values_to_remove = df3['audio_path'].tolist() 

# print(values_to_remove)
# sys.exit()
df1_filtered = df1[df1['audio_path'].isin(values_to_remove)]
print(df1_filtered.shape)
df2_filtered = df2[df2['audio_path'].isin(values_to_remove)]
print(df2_filtered.shape)

df_final = pd.concat([df1_filtered, df2_filtered], ignore_index = True)
print(df_final.shape)
df_final.to_csv('us_sglwrds.csv', index=False)

# print(df_a_filtered)