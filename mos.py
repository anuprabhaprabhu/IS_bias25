import os, sys
import pandas as pd

folder_path = '/home/anuprabha/Desktop/IS_25_biases/sharedforMOS/mos_scores'


spkr_mos_map = {'spkr1':'CF02','spkr2' : 'CM05' , 'spkr3':'CM08' , 'spkr4':'F02' ,'spkr5': 'F03' , 
                   'spkr6' : 'F04' ,  'spkr7':'F05',  'spkr8':'M01', 'spkr9':'M05', 'spkr10':'M07', 
                   'spkr11' : 'M10' , 'spkr12': 'M11', 'spkr13':'M12', 'spkr14' : 'M14','spkr15': 'M16',
                   'spkr16' : 'F01',  'spkr17':'F03', 'spkr18':'F04', 'spkr19':'FC03', 'spkr20':'M01', 
                   'spkr21' : 'M02', 'spkr22' : 'M03', 'spkr23' : 'M04', 'spkr24' : 'M05', 
                   'spkr25' : 'MC03', 'spkr26':'MC04' }


database_mos_map = {'spkr1':'ua','spkr2' : 'ua' , 'spkr3':'ua' , 'spkr4':'ua' ,'spkr5': 'ua' , 
                   'spkr6' : 'ua' ,  'spkr7':'ua',  'spkr8':'ua', 'spkr9':'ua', 'spkr10':'ua', 
                   'spkr11' : 'ua' , 'spkr12': 'ua', 'spkr13':'ua', 'spkr14' : 'ua','spkr15': 'ua',
                   'spkr16' : 'torgo',  'spkr17':'torgo', 'spkr18':'torgo', 'spkr19':'torgo', 'spkr20':'torgo', 
                   'spkr21' : 'torgo', 'spkr22' : 'torgo', 'spkr23' : 'torgo', 'spkr24' : 'torgo', 
                   'spkr25' : 'torgo', 'spkr26':'torgo'}

severi_mos_map = {'spkr1':'healthy','spkr2' : 'healthy' , 'spkr3':'healthy' , 'spkr4':'mid' ,'spkr5': 'high' , 
                   'spkr6' : 'low' ,  'spkr7':'very low',  'spkr8':'high', 'spkr9':'low', 'spkr10':'mid', 
                   'spkr11' : 'very low' , 'spkr12': 'low', 'spkr13':'high', 'spkr14' : 'very low','spkr15': 'mid',
                   'spkr16' : 'low',  'spkr17':'very low', 'spkr18':'very low', 'spkr19':'healthy', 'spkr20':'mid', 
                   'spkr21' : 'mid', 'spkr22' : 'very low', 'spkr23' : 'mid', 'spkr24' : 'low', 
                   'spkr25' : 'healthy', 'spkr26':'healthy' }

excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
excel_files = ['MOS_sheet_all.xlsx']
# excel_files = ['MOS_kp.xlsx']
print(excel_files)
# sys.exit()
dfs = []

for file in excel_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)
final_df = final_df.dropna()
# final_df['Speaker - files'] = final_df['Speaker - files'].astype(str)

final_df['spkrs_mos'] = final_df['Speaker - files'].apply(lambda x:x.split('_')[0]) #.map(spkr_mos_map)
final_df['spkrs'] = final_df['spkrs_mos'].replace(spkr_mos_map)
final_df['database'] = final_df['spkrs_mos'].replace(database_mos_map)
final_df['severi'] = final_df['spkrs_mos'].replace(severi_mos_map)
# print(final_df['spkrs'].unique())
# print(final_df.head(5))
# print(final_df.shape)
df_ua = final_df[final_df['database']=='torgo']
# print(df_ua.shape)
ua_mos = df_ua.groupby(['severi'])[['Criterion 1', 'Criterion 2']].agg(['mean', 'std']).reset_index()
print(ua_mos)