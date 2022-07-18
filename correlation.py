import pandas as pd
import matplotlib.pyplot as plt
from config import *
from sklearn.decomposition import PCA
from numpy.random import choice


asc_df_ori=pd.read_csv(proj_path+'/data/asc_data.csv', index_col=0)
asc_df=asc_df_ori.drop(['covid','healed','patient id','date','id'],axis='columns')
# print(asc_df)

pca=PCA(n_components=2)
pca.fit(asc_df)
X=pca.transform(asc_df)
print(pca.explained_variance_ratio_)
print (pca.explained_variance_)
print (pca.n_components_)
result_df=pd.DataFrame(X)
result_df.index=asc_df.index
result_df['patient id']=asc_df_ori['patient id']
result_df['date']=asc_df_ori['date']
result_df['covid']=asc_df_ori['covid']
result_df['healed']=asc_df_ori['healed']
result_df.dropna(how='any',inplace=True)
# print(result_df)

plt.figure()
for patient_id, df in result_df.groupby('patient id'):
    if len(df)>1:
        print(patient_id)
        print(df)
        plt.scatter(df.loc[:,0],df.loc[:,1],label=f"{df['patient id'][0]}", marker=choice(markers))
plt.legend()
plt.show()

# plt.figure(figsize=(10,10))
# for date, df in result_df.groupby('date'):
#     if len(df)>1:
#         print(df)
#         plt.scatter(df.loc[:, 0], df.loc[:, 1], label=f"{df['date'][0]}", marker=choice(markers))
# plt.legend()
# plt.show()
