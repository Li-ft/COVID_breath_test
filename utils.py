import sys
from datetime import datetime
from typing import Union

import numpy as np
import yaml
import pandas as pd
import os

month_dic = {
    'jen': '01',
    'feb': '02',
    'mar': '03',
    'apr': '04',
    'may': '05',
    'jun': '06',
    'jul': '07',
    'aug': '08',
    'sep': '09',
    'oct': '10',
    'nov': '11',
    'dec': '12'
}

amu_range = {
    0: (10, 51),
    1: (10, 51),
    2: (49, 151),
    3: (149, 251),
    4: (249, 351)
}


def read_config(config_path):
    with open(config_path, 'rb') as f:
        date = yaml.safe_load_all(f)
        return list(date)


def get_all_file_path(path: str):
    all_path = []
    all_name = []
    all_file_list = os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
    for file in all_file_list:
        file_path = os.path.join(path, file)
        # 如果是文件夹，递归调用函数
        if os.path.isdir(file_path):
            child_paths, child_names = get_all_file_path(file_path)
            all_path.extend(child_paths)
            all_name.extend(child_names)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(file_path):
            all_path.append(file_path)
            all_name.append(file)
    return all_path, all_name


def new_filename(file_name):
    # input example: feb 22 2022 7_4.ASC
    # replace month name with number
    new_file_name = ' '.join([month_dic.get(i, i) for i in file_name.split(" ")])
    splits = new_file_name.split()
    # 日期小于10的时候,在前面加个0
    try:
        if int(splits[1]) < 10:
            new_file_name = splits[2] + splits[0] + '0' + splits[1] + "_" + splits[-1]
        else:
            new_file_name = splits[2] + splits[0] + splits[1] + "_" + splits[-1]
    except ValueError:
        print(f'file name: {file_name}')
        print(f'new file name: {new_file_name}')
        print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%{new_file_name}')

    # 只要文件名不要扩展名
    if '.' in new_file_name: new_file_name = new_file_name.split(".")[0]
    return new_file_name  # 20220222_7_4


def asc_2df(file_path: str) -> pd.DataFrame:
    block = -1  # new scan
    scan_times = []  # Interval of Scan
    amu_nums = []  # amu values (same for all scan)
    amu_values = []  # read values
    lst = []  # save the sequence to create the dataframe
    last_index = -1  # check for last element (new scan)

    scan_times.append('amu')  # st fixed name

    with open(file_path, "r+") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()  # ex file jul 9 2021 1_2.ASC contains lines with a space at the beginning

            if line_idx <= 12:  # print file information 12行以内是文件信息
                # print(line)
                continue

            # 在第13行之后任何出现空行的时候 就是一个block被扫描完的时候
            if line.strip() == "" or line in ['\n', '\r\n']:  # white line = end of a scan

                if block == 0:  # first block scan finished: save indexes and values (both series)
                    last_index = amu_nums[-1]
                    lst.append(pd.Series(amu_nums))
                    lst.append(pd.Series(amu_values))
                    amu_values.clear()  # very imp to vlear the values!
                continue

            # 每一个block的开头都有一个等号
            if "=" in line:  # Interval of Scan line -> save the name
                *_, scan_time = line.split('=')
                scan_times.append(scan_time.rstrip())
                block = block + 1  # new block
                continue

            # common value line
            line = line.rstrip()
            amu_num, *_, amu_value = line.split(' ')
            amu_nums.append(float(amu_num))
            amu_values.append(int(amu_value))

            if float(amu_num) == last_index:  # end -> last element of a block
                lst.append(pd.Series(amu_values))
                amu_values.clear()

    # df creation from all series
    df = pd.concat(lst, axis=1)

    # print(scan_time)
    df.columns = scan_times

    df = df.set_index('amu')
    # 处理溢出值 溢出值一般为负数
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x if x >= 0 else 2 ** 32 - 1 + x)
    return df


def most_correlated_columns(df: pd.DataFrame, n: int, filename=None) -> list:
    corr_matrix = df.corr().abs()

    # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
           .stack()
           .sort_values(ascending=False))

    # first element of sol series is the pair with the biggest correlation
    # 把每个变量的最相关的两个变量加进下面这个列表里
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


def feature_selection_corrmax(df: pd.DataFrame, file_path: str, n: int, norm):
    file_name = new_filename(os.path.basename(file_path))  # obtained file name: 20210721_3_1.ASC
    # remove all 0 columns
    # df = df.replace([np.inf, -np.inf, np.nan], 0)
    df = df.loc[:, (df != 0).any()]

    # Normalization bu sum!
    #     col = df.sum(axis=0).argmax() # return index of max sum
    #     df= df / df.sum(axis=0)
    #
    #
    # Standard Scaler: z = (x - u) / s
    #    df = (df - df.mean(axis=0)) / df.std(axis=0)

    original_df = df.copy()

    if norm == 'ns':
        col = df.sum(axis=0).argmax()  # return index of max sum
        df = df / df.sum(axis=0)
    if norm == 'std':
        df = (df - df.mean(axis=0)) / df.std(axis=0)

    list_corr_ori = most_correlated_columns(original_df, n, file_name)
    list_corr = most_correlated_columns(df, n)
    # 每个amu的中位数
    highest_or = original_df.loc[:, list_corr_ori].median(axis=1)
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
    splits = file_name.split("_")
    highest['day'] = highest_or['day'] = file_name.split('_')[0]
    highest['day_pat'] = highest_or['day_pat'] = file_name.split('.')[0].split("_")[0] + "_" + \
                                                 file_name.split('.')[0].split("_")[
                                                     1]
    highest['num'] = highest_or['num'] = file_name.split('.')[0].split("_")[-1]
    df_h.loc[0] = highest
    df_h_o.loc[0] = highest_or

    # print(df_h_o)

    return df_h, df_h_o


def read_asc_files(path: str) -> tuple[list, list]:
    list_ori = []
    list_norm = []
    file_name = os.path.basename(path)
    # print("file_name "+file_name)
    if '_' in file_name:
        new_file_name = new_filename(file_name)  # 20220222_7_4

        ################################
        ####### SELECT THE RANGE #######
        ################################

        if new_file_name.split('_')[-1] == '3':  # NUMBER OF RANGE!!!!
            df = asc_2df(path)
            features_n, features_o = feature_selection_corrmax(df, path, 3, 'ns')
            # break
            # print(features_o)
            if count == 0:
                df_o = pd.DataFrame([], columns=list(features_o.columns))
                df_n = pd.DataFrame([], columns=list(features_n.columns))
                count += 1
            df_o.loc[new_file_name] = features_o.loc[0]
            df_n.loc[new_file_name] = features_n.loc[0]
            # print("added")
    return list_ori, list_norm


def read_all_files(paths):
    count = 0
    patients_df = pd.DataFrame(columns=['covid', 'healed'])
    ls_ori = []
    ls_norm = []

    for path in paths:
        _, ext = os.path.splitext(path)
        # print("path "+path)
        if path.endswith('.ASC'):
            print(path)

        if ext == '.ASC':
            # the paths are like     NTA\Raffaele Correale - feb 22 2022 7_4.ASC
            list_ori, list_norm = read_asc_files(path)
            ls_ori.extend(list_ori)
            ls_norm.extend(list_norm)

        elif ext == '.csv':
            # get the df using pd.readcsv
            patients = pd.read_csv(path, sep=';', ).dropna(how='all')
            # print(patients.columns)
            patients['numP'] = patients['# File'].map(lambda x: x.split('_')[0])
            patients['Data Test'] = patients['Data Test'].map(
                lambda x: x.split('/')[2] + x.split('/')[1] + x.split('/')[0])
            # print(patients)
            for numP, group in patients.groupby('numP'):
                newp = str(list(group['Data Test'])[0]) + "_" + str(numP)
                is_covid = list(group[group.columns[4]])[-1]
                healed = list(group['Guarito'])[-1]
                if str(is_covid) != 'nan':
                    patients_df.loc[newp, 'covid'] = is_covid
                    patients_df.loc[newp, 'healed'] = healed





        elif ext == '.xlsx':
            # get the df using pd.read_excel
            try:
                patients = pd.read_excel(path,
                                         header=0).dropna(thresh=6)
            except Exception:
                continue

            if patients.columns[0] != 'Data Test':
                patients = pd.read_excel(path,
                                         header=1).dropna(thresh=6)
            try:
                patients['Data Test'] = patients['Data Test'].apply(lambda date: date.strftime('%Y/%m/%d'))
            except Exception:
                patients['Data Test'] = patients['Data Test'].apply(lambda date: ''.join(date.split('/')))

            # 1_2 -> 1
            try:
                patients['numP'] = patients['# File'].map(lambda x: x.split('_')[0])
            except Exception as e:
                print(patients)
                print(e)
                print(patients.columns)
                sys.exit('error%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            # try:
            #     patients['Data Test'] = patients['Data Test'].map(
            #         lambda x: (str(x).split('-')[0] + str(x).split('-')[1] + str(x).split('-')[2]).split(" ")[0])
            # except IndexError:
            #     patients['Data Test'] = patients['Data Test'].map(lambda x: print(str(x).split('-')))
            # print(patients)
            # check same patient lines
            for numP, group in patients.groupby('numP'):

                newp = str(list(group['Data Test'])[0]) + "_" + str(numP)  # '20210714_9'
                is_covid = list(group[group.columns[4]])[-1]
                healed = list(group['Guarito'])[-1]
                if str(is_covid) != 'nan':
                    patients_df.loc[newp, 'covid'] = is_covid
                    patients_df.loc[newp, 'healed'] = healed
