import pandas as pd
from config import *
from utils import *
from plot import pp

paths,names=get_all_file_path(data_dir_path)
patients_df, asc_data_df=read_all_files(paths,range_idx, most_core_rec_num, mode)

print(f'num of patients: {len(patients_df)}')
print(f'num of records: {len(asc_data_df)}')

# extract the records of air sample
asc_data_df['index']=asc_data_df.index
asc_data_air_df=asc_data_df[asc_data_df['index'].str.contains('_0')]
asc_data_df=asc_data_df[~asc_data_df['index'].str.contains('_0')]

# get the date and id from the index
asc_data_df['date']=asc_data_df['index'].apply(lambda x: x.split('_')[0])
asc_data_df['id']=asc_data_df['index'].apply(lambda x: x.split('_')[1])
asc_data_df['id']=pd.to_numeric(asc_data_df['id'])
# asc_data_df=asc_data_df[['index','date','id']]

# reformat the index
asc_data_df['index']=asc_data_df['index'].apply(lambda x: '_'.join(x.split('_')[:2]))
asc_data_df.set_index(['index'],inplace=True)
asc_data_air_df.set_index(['index'],inplace=True)
asc_data_df.sort_values(by=['date','id'],inplace=True)

common_idx=set(asc_data_df.index).intersection(set(patients_df.index))
asc_data_df.loc[common_idx, ['covid','healed','patient id']]=patients_df.loc[common_idx]
print(asc_data_df.drop(common_idx))
# asc_data_df.drop(columns=['id','date'],inplace=True)

print(pd.value_counts(asc_data_df.index, sort=True))
print(pd.value_counts(patients_df.index, sort=True))

print(asc_data_df.loc['20210618_2'])
asc_data_df.to_csv(proj_path+f'/data/patient_sample range {range_idx}.csv')
asc_data_air_df.to_csv(proj_path+'/data/air_sample.csv')
patients_df.to_csv(proj_path+'/data/patients_info.csv')
# print(set(patients_df.index).difference(set(asc_data_df.index)))
pp()