import pandas as pd
from config import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# yaml_path=r'./configuration.yml'
# print(proj_path)
features=['30.0','44.0']
# patient_info_df=pd.read_csv(proj_path+'/data/patients_info.csv',index_col=0)
asc_df=pd.read_csv(proj_path+'/data/asc_data.csv', index_col=0)
# print(pd.value_counts(patient_info_df['healed']))
# print(pd.value_counts(patient_info_df['covid']))
asc_df.dropna(subset='covid',inplace=True)
asc_df['index']=asc_df.index
asc_df.index=range(len(asc_df))

num_data=len(asc_df)
num_train=int(num_data*0.7)
num_test=num_data-num_train
print(num_train, num_test, num_data)

asc_df=asc_df.sample(frac=1)

# print(asc_df.index)
# print(asc_df[features])
pos_asc_df=asc_df[asc_df['covid']==1]
neg_asc_df=asc_df[asc_df['covid']==0]
# print(pos_asc_df)

plt.figure()
plt.scatter(pos_asc_df['30.0'],pos_asc_df['44.0'],label='pos')
plt.scatter(neg_asc_df['30.0'],neg_asc_df['44.0'],label='neg')
plt.legend()
# plt.show()

# asc_df
x_train,x_test=asc_df.loc[:num_train,features],asc_df.loc[num_train:,features]
y_train,y_test=asc_df.loc[:num_train, 'covid'],asc_df.loc[num_train:, 'covid']
svm=SVC()
svm.fit(x_train,y_train)

pred_result=svm.predict(x_test)

i=0
for result, pred in zip(y_test,pred_result):
    if result != pred:
        i+=1
print(i/num_test)