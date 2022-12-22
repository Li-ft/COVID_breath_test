from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from sklearn.decomposition import PCA
from numpy.random import choice
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

patient_sample_df=pd.read_csv(proj_path + '/data/patient_sample range 2.csv', index_col=0)
patient_sample_df.drop(columns=['patient id','date','id','healed'],inplace=True)
print(patient_sample_df.head(10))

# feat_cols=list(patient_sample_df.loc[:,::5].columns)
feat_cols=list(patient_sample_df.columns)

feat_cols.append('covid')
print(f'feat cols: {feat_cols}')

asc_data_df=patient_sample_df.drop(columns=['covid'])
# asc_data_df=asc_data_df.div(asc_data_df.columns.values.astype(float), axis=1)
asc_data_df=normalize(asc_data_df.copy(),norm_mode='ns')
patient_sample_df.loc[asc_data_df.index, asc_data_df.columns]=asc_data_df
print(patient_sample_df.head(10))

cls_w = compute_class_weight('balanced',
                             classes=np.unique(patient_sample_df['covid']),
                             y=patient_sample_df['covid'])
print(f'class weights: {cls_w}')

accuracy_result=[]
recall_result=[]
precision_result=[]
f1_result=[]

for i in range(20):
    print(f'iter {i}')
    x_train, x_test, y_train, y_test = train_test_split(patient_sample_df.drop(columns=['covid']),
                                                        patient_sample_df['covid'],
                                                        shuffle=False,
                                                        test_size=0.3)
    clf1 = KNeighborsClassifier(n_neighbors=7)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    # clf3=LinearDiscriminantAnalysis()
    clf4 = GradientBoostingClassifier(random_state=1)
    clf5 = SVC(gamma='auto')
    eclf = VotingClassifier(estimators=[('knn', clf1), ('rf', clf2), ('gnb', clf3), ('gbc', clf4), ('svc', clf5)],
                            voting='hard')
    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf],
                          ['KNN', 'Random Forest', 'Naive Bayes', 'Gradient Boosting', 'SVC', 'Ensemble']):
        scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    eclf.fit(x_train, y_train)
    y_pred = eclf.predict(x_test)
    print(f'bagging score: {eclf.score(x_test, y_test)}')
    # true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test, y_pred).ravel()
    # print(f'true neg: {true_neg}, false pos: {false_pos}, false neg: {false_neg}, true pos: {true_pos}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'f1 score: {f1_score(y_test, y_pred)}\n\n')
    accuracy_result.append(accuracy_score(y_test, y_pred))
    recall_result.append(recall_score(y_test, y_pred))
    precision_result.append(precision_score(y_test, y_pred))
    f1_result.append(f1_score(y_test, y_pred))

print(f'accuracy: {np.mean(accuracy_result)}')
print(f'recall: {np.mean(recall_result)}')
print(f'precision: {np.mean(precision_result)}')
print(f'f1 score: {np.mean(f1_result)}')
