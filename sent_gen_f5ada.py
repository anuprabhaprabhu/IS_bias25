import os, sys
import subprocess
import pandas as pd
import random

def generate_toml_and_run_infer(wav_file):
    # Define the contents of the TOML file
    # toml_content = f"""
    # model = X
    # ref_audio = "{wav_file}"
    # ref_text = "Some call me nature, others call me mother nature."
    # gen_text = "I don't really care what you call me. I've been a silent spectator"
    # output_dir = "{output_dir}"
    # """
    wav_file = row['audio_path']
    ref_text = row['text']
    gen_text = row['gen_text']
    output_file = row['out_path']
    # toml_content = f"""
    # model = "F5-TTS"
    # ref_audio = "{wav_file}"
    # # If an empty "", transcribes the reference audio automatically.
    # ref_text = "{ref_text}"
    # gen_text = "{gen_text}"
    # # File with text to generate. Ignores the text above.
    # #gen_file = ""
    # remove_silence = false
    # output_dir = "/home2/anuprabha.m/dysar_25/F5-TTS/src/f5_tts/infer/gen_combined_ua/"
    # output_file = "{output_file}"
    # """

    # toml_file = f"{os.path.basename(wav_file).replace('.wav','')}.toml"
    # with open(toml_file, 'w') as f:
    #     f.write(toml_content)

    # subprocess.run(["f5-tts_infer-cli", "-c", toml_file])
    # os.remove(toml_file)

    command = [
        'python', 'infer_cli.py', 
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

for idx,spkr in enumerate(uni_spkrs):
    print(idx, spkr)
    temp_df = df[df['audio_path'].apply(lambda x: os.path.basename(x).split('_')[0]) == uni_spkrs[idx]]
    for _, row in temp_df.iterrows():
        print(row)
        generate_toml_and_run_infer(row)
        sys.exit()
