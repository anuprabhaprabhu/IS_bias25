import os, sys
import numpy as np
import pandas as pd 


csv_pth = sys.argv[1]
train_pth = sys.argv[2]
val_pth = sys.argv[3]

df = pd.read_csv(csv_pth)
unispkrs = ['M04','M01','F04' ,'M03','M05', 'FC03', 'MC03']
print(df.shape)
df = df[df['audio_path'].apply(lambda x: x.split('/')[1]).isin(unispkrs)]
print(df.shape)

train_audiopth = pd.DataFrame()
train_text = pd.DataFrame()
val_audiopth = pd.DataFrame()
val_text = pd.DataFrame()

# df_audiopth = pd.DataFrame(df['audio_path'])
# df_audiopth['filename'] = df_audiopth['audio_path'].apply(lambda x: x.split('/')[1])
# df_audiopth['path'] = df_audiopth['audio_path'].apply(lambda x: '/content/drive/MyDrive/is25_dysarASR/' +str(x))
# df_audiopth = df_audiopth.drop('audio_path', axis=1)
# print(df_audiopth.head(5))

# df_text = pd.DataFrame(df_audiopth['filename'])
# df_text['text'] = df['text']
# print(df_text.head(5))
# sys.exit()

# df_audiopth.to_csv('audio_paths', sep=' ', header=False, index=False)
# df_text.to_csv('text', sep=' ', header=False, index=False)
# print('Completted !!!!')

# 10% to deve set 
df = df.drop(['label','gen_text','out_path'], axis=1)
df['new_filename'] = df['audio_path'].str.replace(r'^[^/]+/(M\d+/\d+)\.wav$', r'\1.wav', regex=True)
df['new_filename'] = df['new_filename'].str.replace('/', '_')
# print(df.head(5))
# sys.exit()
for id,spk in enumerate(unispkrs):
    print(id, spk)
    train_df = df[df['audio_path'].apply(lambda x: x.split('/')[1]) == unispkrs[id]]
    
    # train_df['filename'] = train_df['audio_path'].apply(lambda x: x.split('/')[-1].replace('.wav', ''))
    train_df['filename'] = train_df['new_filename'].apply(lambda x: x.replace('.wav', ''))
    train_df['abs_path'] = train_df['new_filename'].apply(lambda x: '/home2/meenakshi.sirigiraju/data/torgo_ref/' +str(x))

    train_df_audiopth = pd.DataFrame(train_df[['filename','abs_path']])
    train_df_text = pd.DataFrame(train_df[['filename','text']])

    # print(train_df_audiopth.head(5))


    # valid - randomly 10% from each speaker
    val_df = train_df.groupby(train_df['audio_path'].apply(lambda x: x.split('/')[0]), group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
    val_df_audiopth = pd.DataFrame(val_df[['filename','abs_path']])
    val_df_text = pd.DataFrame(val_df[['filename','text']])

    train_df_audiopth = train_df_audiopth[~train_df_audiopth.isin(val_df_audiopth)].dropna(how='all')
    train_df_text = train_df_text[~train_df_text.isin(val_df_text)].dropna(how='all')

    train_audiopth = pd.concat([train_audiopth, train_df_audiopth], ignore_index=True)
    train_text = pd.concat([train_text, train_df_text], ignore_index=True)

    val_audiopth = pd.concat([val_audiopth, val_df_audiopth], ignore_index=True)
    val_text = pd.concat([val_text, val_df_text], ignore_index=True)

    

print(train_audiopth.shape)
print(train_text.shape)
print(val_audiopth.shape)
print(val_text.shape)
# train_audiopth = train_audiopth[~train_audiopth.isin(val_audiopth)].dropna(how='all')

# print(train_audiopth.shape)
# sys.exit()

train_audiopth.to_csv(f'{train_pth}/audio_paths', sep=' ', header=False, index=False)
train_text.to_csv(f'{train_pth}/text', sep=' ', header=False, index=False)
val_audiopth.to_csv(f'{val_pth}/audio_paths', sep=' ', header=False, index=False)
val_text.to_csv(f'{val_pth}/text', sep=' ', header=False, index=False)

print("completted!!!")