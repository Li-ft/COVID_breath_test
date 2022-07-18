import pandas as pd
from config import *
from utils import *

# proj_path = os.path.abspath('.')
# yaml_path=r'./configuration.yml'
# config=read_config(yaml_path)

# file_dir=config[0]['data_dir']
# proj_path=config[0]['proj_path']

paths,names=get_all_file_path(data_dir_path)
patients_df, asc_data_df=read_all_files(paths)

# patients_df=pd.read_csv(proj_path+'/data/patients_info.csv',index_col=0)
# asc_data_df=pd.read_csv(proj_path+'/data/asc_data.csv', index_col=0)

print(f'num of patients: {len(patients_df)}')
print(f'num of records: {len(asc_data_df)}')
# patient_day_count=pd.value_counts(pd.Series(patients_df.index).apply(lambda x: x.split('_')[0])).sort_index()
# test2=list(pd.Series(asc_data_df.index).apply(lambda x: x.split('_')[0]))
# for i in test2:

# print(patient_day_count)
# print(patients_df.index)
# for idx in asc_data_df.index:
#     print(idx)
# print(pd.value_counts(asc_data_df.index, sort=True))

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
asc_data_df.sort_values(by=['date','id'],inplace=True)

common_idx=set(asc_data_df.index).intersection(set(patients_df.index))
asc_data_df.loc[common_idx, ['covid','healed','patient id']]=patients_df.loc[common_idx]
print(patients_df.loc['20210618_2'])
print(asc_data_df.loc['20210618_2'])
print('unique patients')
print(asc_data_df.drop(common_idx))
# asc_data_df.drop(columns=['id','date'],inplace=True)

print(pd.value_counts(asc_data_df.index, sort=True))
print(pd.value_counts(patients_df.index, sort=True))

print(asc_data_df.loc['20210618_2'])
asc_data_df.to_csv(proj_path+'/data/patient_sample.csv')
asc_data_air_df.to_csv(proj_path+'/data/air_sample.csv')
patients_df.to_csv(proj_path+'/data/patients_info.csv')
# print(set(patients_df.index).difference(set(asc_data_df.index)))