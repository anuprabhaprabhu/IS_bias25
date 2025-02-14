import os, sys
import numpy as np
import pandas as pd 

csv_pth = sys.argv[1]
test_pth = sys.argv[2]

df = pd.read_csv(csv_pth)
unispkrs = ['M02','F01','F03','MC04']
print(df.shape)
df = df[df['audio_path'].apply(lambda x: x.split('/')[1]).isin(unispkrs)]
print(df.shape)

test_audiopth = pd.DataFrame()
test_text = pd.DataFrame()


df = df.drop(['label','gen_text','out_path'], axis=1)
df['new_filename'] = df['audio_path'].str.replace(r'^[^/]+/(M\d+/\d+)\.wav$', r'\1.wav', regex=True)
df['new_filename'] = df['new_filename'].str.replace('/', '_')
# print(df.head(5))
# sys.exit()
for id,spk in enumerate(unispkrs):
    print(id, spk)
    test_df = df[df['audio_path'].apply(lambda x: x.split('/')[1]) == unispkrs[id]]
    
    # train_df['filename'] = train_df['audio_path'].apply(lambda x: x.split('/')[-1].replace('.wav', ''))
    test_df['filename'] = test_df['new_filename'].apply(lambda x: x.replace('.wav', ''))
    test_df['abs_path'] = test_df['new_filename'].apply(lambda x: '/home2/meenakshi.sirigiraju/data/torgo_ref/' +str(x))

    test_df_audiopth = pd.DataFrame(test_df[['filename','abs_path']])
    test_df_text = pd.DataFrame(test_df[['filename','text']])

    # print(train_df_audiopth.head(5))

    test_audiopth = pd.concat([test_audiopth, test_df_audiopth], ignore_index=True)
    test_text = pd.concat([test_text, test_df_text], ignore_index=True)

print(test_audiopth.shape)
print(test_text.shape)

test_audiopth.to_csv(f'{test_pth}/audio_paths', sep=' ', header=False, index=False)
test_text.to_csv(f'{test_pth}/text', sep=' ', header=False, index=False)

print("completted!!!")