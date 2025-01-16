import os, sys 
import pandas as pd
from pydub import AudioSegment
import random
# # sound1 6 dB louder
# louder = sound1 + 6

fle = sys.argv[1]   # input csv file
out_pth = sys.argv[2]
src_fldr = '/home/anuprabha/Desktop/IS_25_biases/data/ua_speech'

df = pd.read_csv(fle)
print(df.shape)
uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs)

silence_duration = 100  ## 0.5ms
output_data = []
for idx,spkr in enumerate(uni_spkrs):
    print(idx, spkr)
    temp_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    uniq_files = temp_df['audio_path'].tolist() #.apply(lambda x:os.path.basename(x))
    id=1
    unique_sentences = set()
    while len(unique_sentences) < 15:
        num_rows = random.randint(9, 12)
        sent_df = temp_df.sample(n=num_rows)
        # print(sent_df.head(15))

        combined = AudioSegment.empty()
        combined_text = ""

        for index, row in sent_df.iterrows():
            filename = row['audio_path']  
            audio = AudioSegment.from_wav(os.path.join(src_fldr, filename))  
            combined += audio  
            combined += AudioSegment.silent(duration=silence_duration)
            combined_text += row.get('text', '') + " "  

        print(combined_text)
        combined.export(os.path.join(out_pth,f'{spkr}_{id}.wav'), format="wav")
        output_data.append({
            'audio_path': os.path.join(out_pth,f'{spkr}_{id}.wav'),  # The filename of the combined audio
            'text': combined_text.strip()  # Removing any trailing space from the text
        })
        id+=1
        # sys.exit()
        unique_sentences.add(combined_text)

    # # Print the sentences
    # for sentence in unique_sentences:
    #     print(sentence)
    
    # sys.exit()

output_df = pd.DataFrame(output_data)
output_df.to_csv(os.path.join(out_pth,f'ua_sent_15spkrs.csv'), mode='w', header=True, index=False)
print('Completted !!!!')