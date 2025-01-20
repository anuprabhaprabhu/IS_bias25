import torch
import os, sys
import pandas as pd
from transformers import Wav2Vec2Processor, HubertForCTC
import torchaudio
import evaluate
import re

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

csv_file = sys.argv[1] 
fldr_pth = sys.argv[2]

df = pd.read_csv(csv_file)
df['wer'] = None
df['cer'] = None

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for index, row in df.iterrows():
    audio_file = row['out_path']  
    audio_path = os.path.join(fldr_pth, audio_file)  

    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    input_values = processor(waveform.squeeze().numpy(), return_tensors="pt").input_values  # Convert to numpy and process
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    actual_transcription = row['gen_text'] 
    processed_transcription = preprocess_text(transcription)
    processed_actual_transcription = preprocess_text(actual_transcription)

    wer = wer_metric.compute(predictions=[processed_transcription], references=[processed_actual_transcription])
    cer = cer_metric.compute(predictions=[processed_transcription], references=[processed_actual_transcription])

    df.at[index, 'wer'] = wer  
    df.at[index, 'cer'] = cer   
    # print(processed_actual_transcription)
    # print('pred:',processed_transcription)
    print(wer, cer)
    # # print(df.head(5))
    # sys.exit()
df.to_csv('torgo_wer_sim.csv', index=False)

print("Predictions, WER, and CER saved to the CSV file.")