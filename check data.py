import pandas as pd
from config import *
from utils import *
from datetime import datetime

# This check is only valid for the data collected in 2021


patient_detail_df=pd.read_excel(r"C:\Users\maqly\Desktop\patient data detail.xlsx",keep_default_na=False)
patient_detail_df.dropna(subset='18/0',inplace=True)
# print(patient_detail_df['is_covid'])
# print(patient_detail_df.columns)
# print(patient_detail_df.head())
patient_detail_df=patient_detail_df[['18/0','COGNOME','NOME','Data ricovero','Data uscita']]
patient_detail_df['is_covid']=patient_detail_df['18/0'].apply(lambda x: 1 if x.lower()=='p' else 0)
patient_detail_df['full_name']=patient_detail_df['COGNOME']+' ' + patient_detail_df['NOME']
patient_detail_df.drop(columns=['18/0','COGNOME','NOME'],inplace=True)
patient_detail_df.dropna(subset='full_name',inplace=True)
print(len(patient_detail_df.drop_duplicates(subset='full_name')))
print(len(patient_detail_df))
# only useful under current condition (only 2021 records)
patient_detail_df.drop_duplicates(subset='full_name',inplace=True)
patient_detail_df['patient id']=range(len(patient_detail_df))
patient_detail_df.to_csv(proj_path+'/data/test1.csv')

# if len(patient_detail_df.drop_duplicates(subset='full_name'))==len(patient_detail_df):
#     print('yes')
#     patient_detail_df['patient id']=range(len(patient_detail_df))
# else:
#     print(patient_detail_df[patient_detail_df.duplicated(subset='full_name')])

# print(patient_detail_df)

patient_brief_df=pd.read_csv(r"C:\Users\maqly\Desktop\thesis\data\patients_info.csv",index_col=0)
patient_brief_df['index']=patient_brief_df.index
# patient_brief_df.drop(columns='patient id',inplace=True)
patient_brief_df['patient id']=pd.NA
patient_brief_df['index']=patient_brief_df.index
patient_brief_df['date']=patient_brief_df['index'].apply(lambda x: int(x.split('_')[0]))
print(patient_brief_df.columns)
patient_brief_df=patient_brief_df[patient_brief_df['date']<20220101]
patient_brief_df=patient_brief_df[~patient_brief_df['index'].str.contains('_0')]
patient_brief_df.drop(columns=['date'],inplace=True)

for idx,row in patient_brief_df.iterrows():
    df=patient_detail_df[['patient id','full_name']].copy()
    df.loc[:,'name similarity']=df['full_name'].apply(lambda x: string_similar(x.lower(), row['full name'].lower()))
    idx_max=df['name similarity'].idxmax()
    str_similarity_max=df['name similarity'].max()

    # only if the strings are similar enough, the two records can be treated from the same patient
    if str_similarity_max<str_similarity_thresh:
        continue
    # print(f"similarity: {str_similarity_max}")
    row['patient id']=df.loc[idx_max,'patient id']
    # row['full_name']=df.loc[idx_max,'full_name']
    row['similarity']=str_similarity_max
    # print(row)
    # print(row['full_name'],row['full name'])
    patient_brief_df.loc[idx,row.index]=row
    # print(patient_brief_df.loc[idx])
# print(patient_brief_df)
patient_brief_df.to_csv(proj_path+'/data/test2.csv')

out_df=pd.merge(patient_brief_df,patient_detail_df,how='outer',left_on='full name',right_on='full_name')
# print(out_df)
out_df.to_csv(proj_path+'/data/outer join table.csv')

# inner_df=pd.merge(patient_brief_df, patient_detail_df, how='inner', left_on='full name', right_on='full_name')
# inner_df.drop(columns='full name',inplace=True)
# inner_df['check_covid']=inner_df['covid']-inner_df['is_covid']
# inner_df=inner_df[['index','Data ricovero','Data uscita','full_name','covid','is_covid','check_covid']]
# inner_df=inner_df[inner_df['check_covid']!=0]
# inner_df.to_csv(proj_path+'/data/inner join table.csv')

left_df=pd.merge(patient_brief_df, patient_detail_df, how='left', left_on='patient id', right_on='patient id')
left_df['covid check']=left_df['covid']-left_df['is_covid']
left_df=left_df[left_df['covid check']!=0]
left_df=left_df[['index','patient id','full name','full_name','Data ricovero','Data uscita','covid','is_covid','covid check']]
left_df.to_csv(proj_path+'/data/left join table.csv')