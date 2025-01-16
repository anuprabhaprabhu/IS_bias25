import os, sys 
import pandas as pd
from pydub import AudioSegment
import random
# # sound1 6 dB louder
# louder = sound1 + 6

fle = sys.argv[1]   # input csv file
out_pth = sys.argv[2]

df = pd.read_csv(fle)
print(df.shape)
uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()
print(uni_spkrs)

def create_sentence(word_list, min_words=9, max_words=12):
    num_words = random.randint(min_words, max_words)
    selected_words = random.sample(word_list, num_words)
    sentence = ' '.join(selected_words).capitalize() + '.'
    return sentence

for idx,spkr in enumerate(uni_spkrs):
    print(idx, spkr)
    temp_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    words = temp_df['text'].unique().tolist()
    print(type(words))  

    # Generate 15 unique sentences
    unique_sentences = set()
    while len(unique_sentences) < 15:
        sentence = create_sentence(words)
        unique_sentences.add(sentence)

    # Print the sentences
    for sentence in unique_sentences:
        print(sentence)
    
    sys.exit()