import pandas as pd
import matplotlib.pyplot as plt
from config import *
from sklearn.decomposition import PCA
from numpy.random import choice
from mpl_toolkits.mplot3d import Axes3D
from utils import *


patient_sample_df=pd.read_csv(proj_path + '/data/patient sample range 1 refined.csv', index_col=0)
covid_df=pd.read_csv(proj_path+'/data/patients_info.csv',index_col=0)
patient_sample_df.loc[patient_sample_df.index,'covid']=covid_df.loc[patient_sample_df.index,'covid']
# patient_sample_df=pd.read_csv(proj_path + '/data/test.csv', index_col=0)
# asc_df=asc_df_ori.drop(['covid','healed','patient id','date','id'],axis='columns')
patient_sample_df['index']=patient_sample_df.index
patient_sample_df['date']=patient_sample_df['index'].apply(lambda x: x.split('_')[0])
patient_sample_df.loc[patient_sample_df.index, 'covid']=patient_sample_df['covid']
print(patient_sample_df)

pca=PCA(n_components=3)
pca.fit(patient_sample_df)
X=pca.transform(patient_sample_df)
print(pca.explained_variance_ratio_)
print (pca.explained_variance_)
print (pca.n_components_)
result_df=pd.DataFrame(X)
result_df.index=patient_sample_df.index
# result_df['patient id']=asc_df['patient id']
# result_df['date']=patient_sample_df['date']
result_df['covid']=patient_sample_df['covid']
# result_df['healed']=asc_df['healed']
result_df.dropna(how='any',inplace=True)
print(result_df)

svm_classifier(result_df, 'covid')

# plt.figure()
# for patient_id, df in result_df.groupby('patient id'):
#     if len(df)>1:
#         print(patient_id)
#         print(df)
#         plt.scatter(df.loc[:,0],df.loc[:,1],label=f"{df['patient id'][0]}", marker=choice(markers))
# plt.legend()
# plt.show()

# plt.figure(figsize=(10,10))
# for date, df in result_df.groupby('date'):
#     if len(df)>1:
#         print(df)
#         plt.scatter(df.loc[:, 0], df.loc[:, 1], label=f"{df['date'][0]}", marker=choice(markers))
# plt.legend()
# plt.show()

fig=plt.figure()
ax1 = plt.axes(projection='3d')
ax2 = Axes3D(fig,auto_add_to_figure=False)
pos_df=result_df[result_df['covid']==1]
neg_df=result_df[result_df['covid']==0]
ax1.scatter3D(neg_df[0], neg_df[1],neg_df[2], label='neg')
ax1.scatter3D(pos_df[0], pos_df[1],pos_df[2], label='pos')
plt.legend()
plt.show()