from pymcd.mcd import Calculate_MCD
import os, sys
import pandas as pd
import librosa
import soundfile as sf


csv_file = sys.argv[1]
fldr_pth_ref = sys.argv[2]
fldr_pth_gen = sys.argv[3]

df = pd.read_csv(csv_file)
df['mcd_plain'] = None
df['mcd_dtw'] = None
df['mcd_dtw_sl'] = None

# instance of MCD class
# three different modes "plain", "dtw" and "dtw_sl" for the above three MCD metrics
mcd_plain = Calculate_MCD(MCD_mode="plain")
mcd_dtw = Calculate_MCD(MCD_mode="dtw")
mcd_dtw_sl = Calculate_MCD(MCD_mode="dtw_sl")

for index, row in df.iterrows():
    audio_ref = row['audio_path']  
    audio_gen = row['out_path']
    ref = os.path.join(fldr_pth_ref, audio_ref)  
    # wav, fs = librosa.load(ref)
    # sr=22050
    # resampled = librosa.resample(wav, orig_sr=fs, target_sr=22050)
    # sf.write(ref, resampled, sr)

    gen = os.path.join(fldr_pth_gen, audio_gen)
    # wav, fs = librosa.load(gen)
    # resampled = librosa.resample(wav, orig_sr=fs, target_sr=22050)
    # sf.write(gen, resampled, sr)
    # print(ref)
    # print(gen)
    ### resampling is neeedeed??????

    mcd_plain_out = mcd_plain.calculate_mcd(ref, gen)
    # print(mcd_plain_out)
    mcd_dtw_out = mcd_dtw.calculate_mcd(ref, gen)
    # print(mcd_dtw_out)
    mcd_dtw_sl_out = mcd_dtw_sl.calculate_mcd(ref, gen)
    print(mcd_plain_out, mcd_dtw_out, mcd_dtw_sl_out)
    
    df.at[index,'mcd_plain'] = mcd_plain_out
    df.at[index,'mcd_dtw'] = mcd_dtw_out
    df.at[index,'mcd_dtw_sl'] = mcd_dtw_sl_out
    # print(df.head(5))
    # sys.exit()
df.to_csv('torgo_wer_sim_mcd.csv', index=False)
print('Completted !!!!!!!!!')