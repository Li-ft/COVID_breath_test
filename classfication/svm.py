import pandas as pd
from config import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from utils import *


# features=['16.0','28.0','40.0','48.0','covid']
# patient_info_df=pd.read_csv(proj_path+'/data/patients_info.csv',index_col=0)
data_df=pd.read_csv(proj_path + '/data/patient sample range 1 refined.csv', index_col=0)
covid_df=pd.read_csv(proj_path+'/data/patients_info.csv',index_col=0)
data_df.loc[data_df.index,'covid']=covid_df.loc[data_df.index,'covid']
# print(pd.value_counts(patient_info_df['healed']))
# print(pd.value_counts(patient_info_df['covid']))






svm_classifier(data_df, 'covid')