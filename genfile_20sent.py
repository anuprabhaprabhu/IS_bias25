import os, sys
import pandas as pd 
import random 

with open('/home/anuprabha/Desktop/IS_25_biases/data/ua_speech/e2sent_togene.txt', 'r') as file:
    lines = [line.strip() for line in file.readlines()]

df = pd.read_csv('/home/anuprabha/Desktop/IS_25_biases/data/torgo/torgo_15perspkr_2class.csv')

# uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[1]).unique()
uni_spkrs = df['audio_path'].apply(lambda x: x.split(os.sep)[1]).unique()
print(uni_spkrs)

genfile_for20sent = pd.DataFrame()
for idx,spkr in enumerate(uni_spkrs):
    print(idx, spkr)
    # temp_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    temp_df = df[df['audio_path'].apply(lambda x: x.split(os.sep)[1]) == uni_spkrs[idx]]
    rows_to_extend = random.sample(temp_df.index.tolist(), 5)
    extended_rows = temp_df.loc[rows_to_extend]

    # Step 2: Append the extended rows to the DataFrame
    df1 = pd.concat([temp_df, extended_rows], ignore_index=True)
    df1['gen_text'] = lines
    # df1['out_path'] = df1['audio_path'].apply(lambda x: f"{spkr}_{df.index[df['audio_path'] == x].tolist()[0] + 1}.wav")
    df1['out_path'] = df1.index.to_series().apply(lambda idx: f"{spkr}_{idx + 1}_gen.wav")
    genfile_for20sent = pd.concat([genfile_for20sent, df1], ignore_index=True)
    # print(df1.head(25))
    # sys.exit()

genfile_for20sent.to_csv("genfile_torgo_for20sent.csv", sep=',', index=False)
print("Completted!!!!!!")
