import difflib
import sys
from datetime import datetime
from typing import Union, Optional

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


def read_config(config_path: str):
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


def new_filename(file_name: str):
    # input example: feb 22 2022 7_4.ASC
    # replace month name with number
    new_file_name = ' '.join([month_dic.get(i, i) for i in file_name.split(" ")])
    # used for anomaly file name
    if '-' in new_file_name:
        _, new_file_name = new_file_name.split('-')
        new_file_name = new_file_name.lstrip()

    if len(splits := new_file_name.split()) == 4:
        month, day, year, file_id = splits
        if int(day) < 10:
            new_file_name = f'{year}{month}0{day}_{file_id}'
        else:
            new_file_name = f'{year}{month}{day}_{file_id}'
    splits = new_file_name.split()
    # when the date is smaller than 10, add a 0 before it
    # try:
    #     if int(splits[1]) < 10:
    #         new_file_name = splits[2] + splits[0] + '0' + splits[1] + "_" + splits[-1]
    #     else:
    #         new_file_name = splits[2] + splits[0] + splits[1] + "_" + splits[-1]
    # except ValueError:
    #     print(f'file name: {file_name}')
    #     print(f'new file name: {new_file_name}')
    #     print(f'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%{new_file_name}')

    # 只要文件名不要扩展名
    if '.' in new_file_name:
        new_file_name, _ = os.path.splitext(new_file_name)
    return new_file_name  # 20220222_7_4


def asc_2df(asc_file_path: str) -> pd.DataFrame:
    block = -1  # new scan
    scan_times = []  # Interval of Scan
    amu_nums = []  # amu values (same for all scan)
    amu_values = []  # read values
    series_lst = []  # save the sequence to create the dataframe
    last_index = -1  # check for last element (new scan)

    scan_times.append('amu')  # st fixed name

    with open(asc_file_path, "r+") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()  # ex file jul 9 2021 1_2.ASC contains lines with a space at the beginning

            if line_idx <= 12:  # print file information 12行以内是文件信息
                # print(line)
                continue

            # 在第13行之后任何出现空行的时候 就是一个block被扫描完的时候
            if line.strip() == "" or line in ['\n', '\r\n']:  # white line = end of a scan

                if block == 0:  # first block scan finished: save indexes and values (both series)
                    last_index = amu_nums[-1]
                    series_lst.append(pd.Series(amu_nums))
                    series_lst.append(pd.Series(amu_values))
                    amu_values.clear()  # very imp to vlear the values!
                continue

            # there is a = at the beginning of every block
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
                series_lst.append(pd.Series(amu_values))
                amu_values = []

    # df creation from all series
    df = pd.concat(series_lst, axis=1)

    # print(scan_time)
    df.columns = scan_times

    # set the amu number column to be the index
    df = df.set_index('amu')
    # deal with the overflow values
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x if x >= 0 else 2 ** 32 - 1 + x)
    return df


def most_correlated_columns(df: pd.DataFrame, n: int, filename: str = None) -> list:
    corr_matrix = df.corr().abs()

    # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
           .stack()
           .sort_values(ascending=False))

    # first element of sol series is the pair with the biggest correlation
    # 把每个变量的最相关的两个变量加进下面这个列表里
    my_list = []
    for i in range(len(sol)):
        # 等待验证
        # amu_num1, amu_num2, *_=sol.index[i]
        # my_list.append(amu_num1)
        # my_list.append(amu_num2)
        # %%%%%%%%%%%%%%%
        if sol.index[i][0] not in my_list:
            my_list.append(sol.index[i][0])
            if len(my_list) >= len(df.columns):
                break

        if sol.index[i][1] not in my_list:
            my_list.append(sol.index[i][1])
            if len(my_list) >= len(df.columns):
                break
        # %%%%%%%%%%%%%%%

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

    # my_list=sorted(set(my_list), key=my_list.index) # 等待验证
    return my_list[0:n]


def normalize(df, norm_mode: str = 'ns') -> pd.DataFrame:
    if norm_mode == 'ns':
        col = df.sum(axis=0).argmax()  # return index of max sum
        df = df / df.sum(axis=0)
    elif norm_mode == 'std':
        df = (df - df.mean(axis=0)) / df.std(axis=0)
    return df


def get_credible_record(df: pd.DataFrame, file_path: str, n: int, norm_mode: str) -> dict[float, float]:
    # file_name = new_filename(os.path.basename(file_path))  # obtained file name: 20210721_3_1.ASC
    # remove all 0 columns
    df = df.loc[:, (df != 0).any()]

    # Normalization bu sum!
    #     col = df.sum(axis=0).argmax() # return index of max sum
    #     df= df / df.sum(axis=0)
    #
    #
    # Standard Scaler: z = (x - u) / s
    #    df = (df - df.mean(axis=0)) / df.std(axis=0)

    # original_df = df.copy()

    # df = normalize(df, norm_mode)

    # %%%%%%%% feature selection %%%%%%%%%
    # most_corr_cols_ori = most_correlated_columns(original_df, n, file_name)
    most_corr_cols_norm = most_correlated_columns(df, n)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # median of each amu values, among the most correlated scan time
    # median_ori = original_df.loc[:, most_corr_cols_ori].median(axis=1)
    median_norm = df.loc[:, most_corr_cols_norm].median(axis=1)

    # amu_nums_ori = list(median_ori.index)
    # amu_nums_norm = list(median_norm.index)
    # amu_nums_norm.append('day')
    # amu_nums_norm.append('day_test')
    # amu_nums_norm.append('amu_range')
    # amu_nums_ori.append('day')
    # amu_nums_ori.append('day_test')
    # amu_nums_ori.append('amu_range')
    # df_norm = pd.DataFrame([], columns=amu_nums_norm)
    # df_ori = pd.DataFrame([], columns=amu_nums_ori)
    # Add LABEL: 'covid' and 'guarito'
    # filename: 20210721_3_1.ASC
    # pure_file_name, _=os.path.splitext(file_name)
    # date, patient_id, amu_range, *_ = pure_file_name.split("_")
    # median_norm['day'] = median_ori['day'] = date
    # median_norm['day_test'] = median_ori['day_test'] = date + "_" + patient_id
    # median_norm['amu_range'] = median_ori['amu_range'] = amu_range

    # column name is the amu numbers, + day, day_test, amu range
    # df_norm.loc[0] = median_norm
    # df_ori.loc[0] = median_ori

    # print(df_ori)
    return dict(zip(median_norm.index, median_norm))  # , dict(zip(median_ori.index, median_ori))
    # return median_norm.index, median_norm
    # return df_norm, df_ori


def read_asc_file(path: str) -> Optional[dict]:
    file_name = os.path.basename(path)
    # print("file_name "+file_name)
    if '_' in file_name:
        new_file_name = new_filename(file_name)  # 20220222_7_4

        ################################
        ####### SELECT THE RANGE #######
        ################################

        if new_file_name.split('_')[-1] in ['1', '0']:  # NUMBER OF RANGE!!!!
            # columns are scan times
            # indexes are amu numbers
            df = asc_2df(path)
            features_norm = get_credible_record(df, path, 3, 'ns')
            # data1={new_file_name: features_norm}
            # df1=pd.DataFrame(data1,columns=data1.keys())
            # break
            # print(features_ori)

            # df_o = pd.DataFrame([], columns=list(features_ori.columns))
            # df_n = pd.DataFrame([], columns=list(features_norm.columns))
            #
            # df_o.loc[new_file_name] = features_ori.loc[0]
            # df_n.loc[new_file_name] = features_norm.loc[0]
            # return df_o, df_n
            return {new_file_name: features_norm}
    return None


def read_all_files(paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients_df = pd.DataFrame(columns=['covid', 'healed'])
    asc_result_dicts = {}
    # patient_name_id={}

    for path in paths:
        if "Preliminari COVID" in path or 'bis' in path:
            continue
        _, ext = os.path.splitext(path)

        if ext == '.ASC':
            # the paths are like     NTA\Raffaele Correale - feb 22 2022 7_4.ASC
            result_dict = read_asc_file(path)
            if result_dict is None:
                continue
            else:
                asc_result_dicts.update(result_dict)

        elif ext == '.csv':
            # get the df using pd.readcsv
            patients = pd.read_csv(path, dtype='str', sep=';').dropna(how='all')
            if len(patients.columns) < 2:
                patients = pd.read_csv(path, dtype='str', sep=',').dropna(how='all')
            try:
                patients['day record id'] = patients['# File'].map(lambda x: '_'.join(x.split('_')[:-1]))
            except Exception as e:
                print(patients.columns)
                print(e)
                print("error in file: " + path)
                # sys.exit(1)

            patients['Data Test'] = patients['Data Test'].map(
                lambda x: "/".join(x.split('/')[::-1]))
            # lambda x: x.split('/')[2] + x.split('/')[1] + x.split('/')[0])
            # print(patients)
            for record_id, group in patients.groupby('day record id'):
                date = str(list(group['Data Test'])[0])
                date = ''.join(date.split('/'))
                record_id = date + "_" + str(record_id)
                full_name = list(group['Nome Cognome'])[0]
                # if full_name in patient_name_id.keys():
                #     patient_id=patient_name_id[full_name]
                # else:
                #     patient_id=len(patient_name_id)
                #     patient_name_id.update({full_name:patient_id})
                is_covid = list(group[group.columns[4]])[-1]
                if is_covid in ['POS', 'SI']:
                    is_covid = 1
                elif is_covid in ['NO', 'no', 'NEG']:
                    is_covid = 0
                else:
                    is_covid = -1
                healed = list(group['Guarito'])[-1]
                if healed in ['SI', 'si']:
                    healed = 1
                elif healed in ['NO', 'no']:
                    healed = 0
                else:
                    healed = -1
                if is_covid != -1:
                    patients_df.loc[record_id, ['covid', 'healed', 'full name']] = [is_covid, healed, full_name]
                    # patients_df.loc[record_id, ]
                    # patients_df.loc[patient_id, 'covid'] = is_covid
                    # patients_df.loc[patient_id, 'healed'] = healed

        elif ext == '.xlsx':
            # this try block is used to filter those encrypted files
            try:
                patients = pd.read_excel(path, header=0).dropna(thresh=5)
            except Exception:
                continue

            # this block is used in the case of using unnamed column names in some files
            if patients.columns[0] != 'Data Test':
                patients = pd.read_excel(path, header=1).dropna(thresh=5)

            # some date in the files is read as Datetime object while the other files has string date
            try:
                patients['Data Test'] = patients['Data Test'].apply(lambda date: date.strftime('%Y/%m/%d'))
            except Exception:
                patients['Data Test'] = patients['Data Test'].apply(lambda date: ''.join(date.split('/')))

            # 1_2 -> 1
            try:
                patients['day record id'] = patients['# File'].map(lambda x: '_'.join(x.split('_')[:-1]))
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
            for record_id, group in patients.groupby('day record id'):
                date = str(list(group['Data Test'])[0])
                date = ''.join(date.split('/'))
                record_id = date + "_" + str(record_id)  # '20210714_9'
                full_name = list(group['Nome Cognome'])[0]
                # if full_name in patient_name_id.keys():
                #     patient_id=patient_name_id[full_name]
                # else:
                #     patient_id=len(patient_name_id)
                #     patient_name_id.update({full_name:patient_id})
                is_covid = list(group[group.columns[4]])[-1]
                if is_covid in ['POS', 'SI']:
                    is_covid = 1
                elif is_covid in ['NO', 'no', 'NEG']:
                    is_covid = 0
                else:
                    is_covid = -1
                healed = list(group['Guarito'])[-1]
                if healed in ['SI', 'si']:
                    healed = 1
                elif healed in ['NO', 'no', 'No']:
                    healed = 0
                else:
                    healed = -1
                if is_covid != -1:
                    patients_df.loc[record_id, ['covid', 'healed', 'full name']] = [is_covid, healed, full_name]
                    # patients_df.loc[patient_id, 'covid'] = is_covid
                    # patients_df.loc[patient_id, 'healed'] = healed

    amu_data_df = pd.DataFrame(asc_result_dicts, columns=asc_result_dicts.keys())
    return patients_df, amu_data_df.transpose()


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()
