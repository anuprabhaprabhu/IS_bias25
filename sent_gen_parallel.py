import os, sys
import subprocess
import pandas as pd
import random
import concurrent.futures


def process_row(row):
    print(row)
    generate_toml_and_run_infer(row)

    
def generate_toml_and_run_infer(wav_file):
    wav_file = row['audio_path']
    ref_text = row['text']
    gen_text = row['gen_text']
    output_file = row['out_path']
#/home2/anuprabha.m/dysar_25/envs/f5_tts/bin/
    command = [
        '/home2/anuprabha.m/dysar_25/envs/f5_tts/bin/python', 'infer_cli.py', 
        '--model', 'F5-TTS',
        '--ref_audio', wav_file,
        '--ref_text', ref_text,
        '--gen_text', gen_text,
        '--output_dir', "/home2/anuprabha.m/dysar_25/F5-TTS/src/f5_tts/infer/gen_combined_ua/",
        '--output_file', output_file
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)


df = pd.read_csv('/home2/anuprabha.m/dysar_25/F5-TTS/src/f5_tts/infer/genfile_uaspee_for20sent.csv')
uni_spkrs = df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]).unique()

with concurrent.futures.ThreadPoolExecutor() as executor:
    for idx,spkr in enumerate(uni_spkrs):
        print(idx, spkr)
        temp_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
        for _, row in temp_df.iterrows():
            print(row)
            generate_toml_and_run_infer(row)
        # sys.exit()

print("Completted !!!!!")
