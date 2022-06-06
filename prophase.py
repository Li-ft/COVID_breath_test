
import os, glob
import numpy as np
import pandas as pd
import os, sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
import math
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from glob import glob
import os
import matplotlib.pyplot as plt
%matplotlib inline

# Set the style
plt.style.use('fivethirtyeight')
import plotly.graph_objects as go
import plotly.express as px

layout = go.Layout(
    autosize=False,
    width=1300,
    height=500,

    margin=go.layout.Margin(
        l=50,
        r=50,
        b=100,
        t=100,
        pad = 4
    )
)

#%%
month_dic ={
    'jen':'01',
    'feb':'02',
    'mar':'03',
    'apr':'04',
    'may':'05',
    'jun':'06',
    'jul':'07',
    'aug':'08',
    'sep':'09',
    'oct':'10',
    'nov':'11',
    'dec':'12'
}

def new_filename(file):
    # remove month name
    file = ' '.join([month_dic.get(i, i) for i in file.split(" ")])
    #
    splits = file.split()
    # 日期小于10的时候,在前面加个0
    if int(splits[1])<10:
        file = splits[2]+splits[0]+'0'+splits[1]+"_"+splits[-1]
    else:
        file = splits[2]+splits[0]+splits[1]+"_"+splits[-1]

    if '.' in file: file = file.split(".")[0]
    return file # eg: 1_2

#%%
# Given the filename returns the DataFrame
def createDF_fromFile(file_path):
    fo = open(file_path, "r+")
    # print ("Name of the file: ", fo.name)

    dimbatch = -1  # sample dimension
    count = 0  # counter
    block = -1  # new scan
    names = []  # Interval of Scan
    amu_num = []  # amu values (same for all scan)
    amu_values = []  # read values
    series = []  # save the sequence to create the dataframe
    last_index = -1  # check for last element (new scan)

    names.append('amu')  # st fixed name

    for line_idx, line in enumerate(fo):
        line = line.strip()  # ex file jul 9 2021 1_2.ASC contains lines with a space at the beginning

        if line_idx < 12:  # print file information 12行以内是文件信息
            # print(line)
            continue

        if line_idx == 12:  # white line第13行是空行
            test = line
            continue

        if line == test:  # white line = end of a scan
            # first scan, save the dimension
            if dimbatch == -1:
                dimbatch = count
                # print("DIMBATCH "+str(dimbatch))

            count = 0  # count reset

            if block == 0:  # first scan: save indexes and values (both series)
                last_index = amu_num[-1]
                series.append(pd.Series(amu_num))
                series.append(pd.Series(amu_values))
                amu_values.clear()  # very imp to vlear the values!
            continue

        if "=" in line:  # Interval of Scan line -> save the name
            el = line.split('=')
            names.append(str(el[-1].rstrip()))
            block = block + 1  # new block
            continue

        # common value line
        line = line.rstrip()
        el = line.split(' ')
        amu_num.append(float(el[0]))
        amu_values.append(int(el[-1]))
        count = count + 1
        if float(el[0]) == last_index:  # end -> last element of a block
            count = 0
            series.append(pd.Series(amu_values))
            amu_values.clear()

    # df creation from all series
    df = pd.concat(series, axis=1)

    # print(names)
    df.columns = names

    df = df.set_index('amu')
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x if x >= 0 else 2 ** 32 - 1 + x)
    return df

#%%
def feature_selection_corrmax(df, file, n, norm):
    file = new_filename(file.split("/")[-1])  # obtained file name: 20210721_3_1.ASC
    # remove all 0 columns
    # df = df.replace([np.inf, -np.inf, np.nan], 0)
    df = df.loc[:, (df != 0).any()]
    # print("################")
    # print(file)
    # print("################")
    # print(df)
    # .astype(np.uint)
    '''


Normalization bu sum!
    col = df.sum(axis=0).argmax() # return index of max sum
    df= df / df.sum(axis=0)


Standard Scaler: z = (x - u) / s
   df = (df - df.mean(axis=0)) / df.std(axis=0)

 '''

    original_df = df.copy()

    if norm == 'ns':
        col = df.sum(axis=0).argmax()  # return index of max sum
        df = df / df.sum(axis=0)
    if norm == 'std':
        df = (df - df.mean(axis=0)) / df.std(axis=0)

    list_corr_or = most_correlated_columns(original_df, n, file)
    list_corr = most_correlated_columns(df, n)
    highest_or = original_df.loc[:, list_corr_or].median(axis=1)
    highest = df.loc[:, list_corr].median(axis=1)

    lis_o = list(highest_or.index)
    lis = list(highest.index)
    lis.append('day')
    lis.append('day_pat')
    lis.append('num')
    lis_o.append('day')
    lis_o.append('day_pat')
    lis_o.append('num')
    df_h = pd.DataFrame([], columns=lis)
    df_h_o = pd.DataFrame([], columns=lis_o)
    # Add LABEL: 'covid' and 'guarito'
    splits = file.split("_")
    highest['day'] = highest_or['day'] = file.split('_')[0]
    highest['day_pat'] = highest_or['day_pat'] = file.split('.')[0].split("_")[0] + "_" + file.split('.')[0].split("_")[
        1]
    highest['num'] = highest_or['num'] = file.split('.')[0].split("_")[-1]
    df_h.loc[0] = highest
    df_h_o.loc[0] = highest_or

    # print(df_h_o)

    return df_h, df_h_o

#%%

from itertools import combinations
import operator


def most_correlated_columns(df, n, filename=None):
    corr_matrix = df.corr().abs()

    # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)

    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
           .stack()
           .sort_values(ascending=False))

    # first element of sol series is the pair with the biggest correlation

    # print(sol)
    my_list = []
    for i in range(len(sol)):
        if sol.index[i][0] not in my_list:
            my_list.append(sol.index[i][0])
            if len(my_list) >= len(df.columns):
                break

        if sol.index[i][1] not in my_list:
            my_list.append(sol.index[i][1])
            if len(my_list) >= len(df.columns):
                break


    # if (filename != None):
    #     d = dict()
    #     for el in list(combinations(np.arange(0,len(my_list)), 2)):
    #         if(df[my_list[el[0]]]-df[my_list[el[1]]]).sum() == 0:
    #             p=0
    #         else:
    #             U1,p =stats.wilcoxon(df[my_list[el[0]]], df[my_list[el[1]]])
    #         d[el] = p
    #
    #     sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    #     file1 = open("wicoxon4.txt", "a")
    #
    #     file1. write(filename + "\n")
    #     for element in sorted_d:
    #         file1. write(str(element) + "\n")
    #     file1. write("\n\n\n")
    #     file1.close()
    return my_list[0:n]

#%%
# recursive reserch on dir
# FOR EACH FILE

from scipy import stats

files = glob('.....PATH....../FILES/**', recursive=True)
patients_df = pd.DataFrame(columns=['covid', 'healed'])
count = 0

for file in files:
    # print(file)
    if '.' in file:

        ext = file.split('.')[-1]
        # print("file "+file)
        if ext == 'ASC':
            # the paths are like     NTA\Raffaele Correale - feb 22 2022 7_4.ASC
            file_name = file.split('/')[-1]
            # print("namef "+namef)
            if '_' in file_name:
                # print("new_filename "+new_filename(namef))
                new_file_name =  new_filename(file_name)  # namef.split('_')[0] + '_' + namef.split('_')[1]
                # print(ind)

                ################################
                ####### SELECT THE RANGE #######
                ################################

                if new_file_name.split('_')[-1] == '3':  # or ind.split('_')[-1] == '1':# NUMBER OF RANGE!!!!
                    df_file = createDF_fromFile(file)
                    features_n, features_o = feature_selection_corrmax(df_file, file, 3, 'ns')
                    # break
                    # print(features_o)
                    if count == 0:
                        df_o = pd.DataFrame([], columns=list(features_o.columns))
                        df_n = pd.DataFrame([], columns=list(features_n.columns))
                        count += 1
                    df_o.loc[new_file_name] = features_o.loc[0]
                    df_n.loc[new_file_name] = features_n.loc[0]
                    # print("added")
        elif ext == 'csv':
            # get the df using pd.readcsv
            # print(file)
            patients = pd.read_csv(file, sep=';', ).dropna(how='all')
            # print(patients.columns)
            patients['numP'] = patients['# File'].map(lambda x: x.split('_')[0])
            patients['Data Test'] = patients['Data Test'].map(
                lambda x: x.split('/')[2] + x.split('/')[1] + x.split('/')[0])
            # print(patients)
            for key, group in patients.groupby('numP'):
                # print(group.columns)
                # print(list(group[group.columns[4]]))

                newp = str(list(group['Data Test'])[0]) + "_" + str(key)
                cov = list(group[group.columns[4]])[-1]
                guar = list(group['Guarito'])[-1]
                if str(cov) != 'nan':
                    patients_df.loc[newp, 'covid'] = cov
                    patients_df.loc[newp, 'healed'] = guar

            # check same patient lines
        elif ext == 'xlsx':
            # get the df using pd.read_excel
            # print(file)
            patients = pd.read_excel(file).dropna(how='all')
            patients['numP'] = patients['# File'].map(lambda x: x.split('_')[0])
            patients['Data Test'] = patients['Data Test'].map(
                lambda x: (str(x).split('-')[0] + str(x).split('-')[1] + str(x).split('-')[2]).split(" ")[0])
            # print(patients)
            # check same patient lines
            for key, group in patients.groupby('numP'):
                # print(group.columns)
                # print(list(group[group.columns[4]]))

                newp = str(list(group['Data Test'])[0]) + "_" + str(key)
                cov = list(group[group.columns[4]])[-1]
                guar = list(group['Guarito'])[-1]
                if str(cov) != 'nan':
                    patients_df.loc[newp, 'covid'] = cov
                    patients_df.loc[newp, 'healed'] = guar

# df_o = find_integer_amu(df_o)
# df_o = df_o.set_index('day_pat_num')
# df_o
# df_o
