# %%
import pandas as pd
from config import *
from utils import *

# %%
paths,names=get_all_file_path(data_dir_path)
pat_df, asc_data_df=read_all_files(paths, range_idx, most_corr_rec_num, mode)
asc_data_df.fillna(0, inplace=True)

print(f'asc data df:\n{asc_data_df.head(5)}')
asc_data_df=asc_data_df.astype(np.int64)
print(f'num of patients: {len(pat_df)}')
print(f'num of records: {len(asc_data_df)}')
# %%
# divide the samples into two groups: air samples and patient samples
asc_data_df['index']=asc_data_df.index
asc_data_air_df=asc_data_df[asc_data_df['index'].str.contains('_0')]
asc_data_df=asc_data_df[~asc_data_df['index'].str.contains('_0')]
# asc_data_air_df=asc_data_df.loc[asc_data_df.index.str.contains('_0')]
# asc_data_df=asc_data_df.loc[~asc_data_df.index.str.contains('_0')]

# get the date and id from the index
asc_data_df['date']=asc_data_df['index'].apply(lambda x: x.split('_')[0])
asc_data_df['id']=asc_data_df['index'].apply(lambda x: x.split('_')[1])
asc_data_df['id']=pd.to_numeric(asc_data_df['id'])
# reformat the index
asc_data_df['index']=asc_data_df['index'].apply(lambda x: '_'.join(x.split('_')[:2]))
asc_data_df.set_index(['index'],inplace=True)
asc_data_air_df.set_index(['index'],inplace=True)
asc_data_df.sort_values(by=['date','id'],inplace=True)

common_idx=set(asc_data_df.index).intersection(set(pat_df.index))
asc_data_df.loc[common_idx, ['covid','healed','patient id']]=pat_df.loc[common_idx]

asc_data_df.to_csv(proj_path+f'/data/patient_sample range {range_idx}.csv')
asc_data_air_df.to_csv(proj_path+f'/data/air_sample range {range_idx}.csv')
pat_df.to_csv(proj_path + '/data/patients_info.csv')