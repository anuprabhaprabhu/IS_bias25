import os, sys 
import pandas as pd
from pydub import AudioSegment

# # sound1 6 dB louder
# louder = sound1 + 6

fle = sys.argv[1]   # input csv file
out_pth = sys.argv[2]

df = pd.read_csv(fle)
print(df.shape)

uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs)
# uni_spkrs = ['M09','M10','M14','M11','M16','M01','M12']
# df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).isin(uni_spkrs)]
print(uni_spkrs)

# silence_duration = 200  ## 1ms

blk_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[1])== 'B1'] 
digi = ['D0','D1','D2','D3','D4']
digi_df = blk_df[blk_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[2]).isin(digi)]
print(digi_df.shape)
output_data = []
for idx,spkr in enumerate(uni_spkrs):
    combined = AudioSegment.empty()
    combined_text = ""
    temp_df = digi_df[digi_df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    print(temp_df.head(10))
    for index, row in temp_df.iterrows():
        filename = row['audio_path']  
        audio = AudioSegment.from_wav(os.path.join('/home/anuprabha/Desktop/anu_donot_touch/data/', filename))  
        combined += audio  
        # combined += AudioSegment.silent(duration=silence_duration)
        combined_text += row.get('text', '') + " "  

    print(combined_text)
    combined.export(os.path.join(out_pth,f'{spkr}_combned.wav'), format="wav")
    output_data.append({
        'audio_path': os.path.join(out_pth,f'{spkr}_combned.wav'),  # The filename of the combined audio
        'text': combined_text.strip()  # Removing any trailing space from the text
    })
    # sys.exit()
output_df = pd.DataFrame(output_data)
output_df.to_csv(os.path.join(out_pth,f'combined_digi.csv'), mode='w', header=True, index=False)
print('Completted !!!!')
