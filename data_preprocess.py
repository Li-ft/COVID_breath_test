import pandas as pd
from config import *
from utils import *

asc_df_ori = pd.read_csv(proj_path + '/data/patient_sample.csv', index_col=0, dtype={'date': int})
asc_air_df = pd.read_csv(proj_path + '/data/air_sample.csv', index_col=0)

print(len(asc_df_ori))
# delete the records without the covid status
asc_df_ori.dropna(inplace=True, subset='covid', how='any')
print(len(asc_df_ori))

# delete those records that are under the same patient in the same day
asc_df = asc_df_ori.drop_duplicates(subset=['date', 'patient id'], keep='last')
# test1=asc_df.drop_duplicates(subset=['date'],keep='last')
asc_air_df['date'] = asc_air_df['index'].apply(lambda x: int(x.split('_')[0]))
asc_air_df['id'] = asc_air_df['index'].apply(lambda x: int(x.split('_')[1]))
asc_air_df.sort_values(by=['date', 'id'], inplace=True)
asc_air_df.drop(columns=['id'], inplace=True)
asc_air_df.drop_duplicates(subset=['date'], keep='first', inplace=True)
# test2=asc_air_df.drop_duplicates(subset=['date'],keep='first')
# print(test1['date'].sort_values())
# print(test2['date'].sort_values())
# print(set(test1['date'].astype(int)).difference(set(test2['date'].astype(int))))
# print(asc_air_df)
# print(asc_df)

for date, group_df in asc_df.groupby('date'):
    print(date)
    # print(len((asc_air_df['date']==date)==True))
    print(group_df)
    test=asc_air_df[asc_air_df['date'] == date]
    print(test)
    if len(test)>0:
        print(test.iloc[0])
    else:
        print('%%%%%%%%%%%%%%%%%%%%%%%%')

