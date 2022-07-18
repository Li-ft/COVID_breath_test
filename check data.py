import pandas as pd
from config import *
from utils import *
from datetime import datetime


patient_detail_df=pd.read_excel(r"C:\Users\maqly\Desktop\patient data detail.xlsx")
patient_detail_df.dropna(subset='18/0',inplace=True)
# print(patient_detail_df['is_covid'])
# print(patient_detail_df.columns)
# print(patient_detail_df.head())
patient_detail_df=patient_detail_df[['18/0','COGNOME','NOME','Data ricovero','Data uscita']]
patient_detail_df['is_covid']=patient_detail_df['18/0'].apply(lambda x: 1 if x.lower()=='p' else 0)
patient_detail_df['full_name']=patient_detail_df['COGNOME']+' ' + patient_detail_df['NOME']
patient_detail_df.drop(columns=['18/0','COGNOME','NOME'],inplace=True)
print(patient_detail_df)

patient_brief_df=pd.read_csv(r"C:\Users\maqly\Desktop\thesis\data\patients_info.csv",index_col=0)
patient_brief_df['index']=patient_brief_df.index
patient_brief_df.drop(columns='patient id',inplace=True)
print(patient_brief_df)

out_df=pd.merge(patient_brief_df,patient_detail_df,how='outer',left_on='full name',right_on='full_name')
print(out_df)
out_df.to_csv(proj_path+'/data/outer join table.csv')

inner_df=pd.merge(patient_brief_df, patient_detail_df, how='inner', left_on='full name', right_on='full_name')
inner_df.drop(columns='full name',inplace=True)
inner_df['check_covid']=inner_df['covid']-inner_df['is_covid']
inner_df=inner_df[['index','Data ricovero','Data uscita','full_name','covid','is_covid','check_covid']]
inner_df=inner_df[inner_df['check_covid']!=0]
inner_df.to_csv(proj_path+'/data/inner join table.csv')

# left_df=pd.merge(patient_brief_df, patient_detail_df, how='left', left_on='full name', right_on='full_name')
# left_df.to_csv(proj_path+'/data/left join table.csv')