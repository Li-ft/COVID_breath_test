import difflib
import sys
from datetime import datetime
from typing import Union, Optional, Collection
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def get_all_file_path(path: str):
    all_path = []
    all_name = []
    all_file_list = os.listdir(path)
    # walk through all the directory and files in the folder
    for file in all_file_list:
        file_path = os.path.join(path, file)
        # if it is a folder, call the function recursively
        if os.path.isdir(file_path):
            child_paths, child_names = get_all_file_path(file_path)
            all_path.extend(child_paths)
            all_name.extend(child_names)
        # if it is a file, save the file path and file name
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

    # only keep the file name, not the extension
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

            if line_idx <= 12:  # print file information (first 12 lines)
                # print(line)
                continue

            # when a block is scanned, there will be a blank line
            if line.strip() == "" or line in ['\n', '\r\n']:  # white line = end of a scan

                if block == 0:  # first block scan finished: save indexes and values (both series)
                    last_index = amu_nums[-1]
                    series_lst.append(pd.Series(amu_nums))
                    series_lst.append(pd.Series(amu_values))
                    amu_values.clear()  # very imp to clear the values!
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

def get_most_corr_cols(df: pd.DataFrame, n: int, filename: str = None) -> list:
    corr_matrix = df.corr().abs()

    # the matrix is symmetric, so we need to extract upper triangle matrix without diagonal (k = 1)
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
           .stack()
           .sort_values(ascending=False))

    # first element of sol series is the pair with the biggest correlation
    # add the most correlated pair of each variable into the list
    my_list = []
    for i in range(len(sol)):
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
    return my_list[0:n]

def normalize(df, norm_mode: str = 'ns') -> pd.DataFrame:
    if norm_mode == 'ns':
        col = df.sum(axis=0).argmax()  # return index of max sum
        df = df / df.sum(axis=0)
    elif norm_mode == 'std':
        df = (df - df.mean(axis=0)) / df.std(axis=0)
    else:
        raise ValueError("norm_mode must be 'ns' or 'std'")
    return df

def get_credible_rec(df: pd.DataFrame, n: int, mode:str= 'mean') -> dict[float, float]:
    # file_name = new_filename(os.path.basename(file_path))  # obtained file name: 20210721_3_1.ASC
    # remove all 0 columns
    df = df.loc[:, (df != 0).any()]
    most_corr_cols = get_most_corr_cols(df, n)
    if mode=='median':
        median_norm = df.loc[:, most_corr_cols].median(axis=1)
    elif mode=='mean':
        median_norm = df.loc[:, most_corr_cols].mean(axis=1)
    return dict(zip(median_norm.index, median_norm))

def read_asc_file(path: str, range_idx:int=1, most_core_rec_num:int=3, mode:str='mean') -> Optional[dict]:
    if type(range_idx) == int:
        range_idx = str(range_idx)
    file_name = os.path.basename(path)
    if '_' in file_name:
        new_file_name = new_filename(file_name)  # 20220222_7_4

        ################################
        ####### SELECT THE RANGE #######
        ################################

        if new_file_name.split('_')[-1] == range_idx:  # NUMBER OF RANGE!!!!
            # columns are scan times
            # indexes are amu numbers
            print(new_file_name)
            df = asc_2df(path).transpose()

            # get the alignment correction based on peaks
            # peak_distribute=get_peak_distribute(df)
            # accu_peak_height=accumulate_peak_by_shift_window(5,df,peak_distribute)
            # raw_feat= get_raw_feat(7,accu_peak_height)

            # df=rec_alignment_half(df,rec_space=0.1)
            df=rec_alignment(df, rec_space=0.1)
            if df is None:
                return None
            else:
                df=df.transpose()

            features_norm = get_credible_rec(df, most_core_rec_num, mode)
            return {new_file_name: features_norm}
    return None

def read_all_files(paths:str,range_idx:int=1,most_core_rec_num:int=3, mode:str='mean') -> tuple[pd.DataFrame, pd.DataFrame]:
    patients_df = pd.DataFrame(columns=['covid', 'healed'])
    asc_result_dicts = {}
    # patient_name_id={}

    for path in paths:
        # skip this path because files under this path are duplicated
        if "Preliminari COVID" in path or 'bis' in path:
            continue
        _, ext = os.path.splitext(path)

        if ext == '.ASC':
            # the paths are like     NTA\Raffaele Correale - feb 22 2022 7_4.ASC
            try:
                result_dict = read_asc_file(path, range_idx, most_core_rec_num, mode)
            except TypeError:
                continue

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

            # check same patient lines
            for record_id, group in patients.groupby('day record id'):
                date = str(list(group['Data Test'])[0])
                date = ''.join(date.split('/'))
                record_id = date + "_" + str(record_id)  # '20210714_9'
                full_name = list(group['Nome Cognome'])[0]

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

    amu_data_df = pd.DataFrame(asc_result_dicts, columns=asc_result_dicts.keys())
    return patients_df, amu_data_df.transpose()

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def svm_classifier(data_df: pd.DataFrame, y_col_name:str):
    data_df.dropna(subset=y_col_name, inplace=True)
    # data_df.drop(columns=['index'], inplace=True)
    # data_df['index'] = data_df.index
    data_df.index = range(len(data_df))

    x_train, x_test, y_train, y_test = train_test_split(data_df.drop(columns=[y_col_name]),
                                                        data_df[y_col_name],
                                                        test_size=0.3)

    pca = PCA(n_components=2)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    # print(f'x train:\n{x_train}')
    x_test= pca.transform(x_test)
    # print(f'x test:\n{x_test}')

    # plot the points from positive and negative patients
    plt.figure(figsize=(30, 20))
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
    plt.ylim(-0.009,0)
    plt.show()

    ## 3d plot of the points
    # fig=plt.figure()
    # ax1 = plt.axes(projection='3d')
    # ax2 = Axes3D(fig,auto_add_to_figure=False)
    # ax1.scatter3D(x_train[:, 0], x_train[:, 1], x_train[:, 2], c=y_train)
    # ax2.scatter3D(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=y_test)
    # plt.legend()
    # plt.show()

    # plot the points from train and test data
    # plt.figure()
    # plt.scatter(x_train[:, 0], x_train[:, 1], label='train')
    # plt.scatter(x_test[:, 0], x_test[:, 1], label='test')
    # plt.legend()
    # plt.show()

    svm=SVC()
    svm.fit(x_train,y_train)

    print(f'x test length: {len(x_test)}')
    pred_result=svm.predict(x_test)

    print(f'classification report:\n{classification_report(y_test,pred_result)}')
    print(f'Accuracy: {accuracy_score(pred_result,y_test)}')
    print(f'Precision: {precision_score(pred_result,y_test)}')
    print(f'Recall: {recall_score(pred_result,y_test)}')
    print(f'f1 score: {f1_score(pred_result,y_test)}')

    true_neg,false_pos,false_neg,true_pos=confusion_matrix(y_test,pred_result).ravel()
    print(f'true pos: {true_pos}')
    print(f'true neg: {true_neg}')
    print(f'false neg: {false_neg}')
    print(f'false pos: {false_pos}')

    # # only plot when data are 2d, another 1 means the 'covid' column
    # if len(data_df.columns)>=2+3:
    #     plt.figure(figsize=(15, 10))
    #     pos_df = data_df[data_df['covid'] == 1]
    #     neg_df = data_df[data_df['covid'] == 0]
    #     print(f'pos df:\n{pos_df}')
    #     print(f'neg df:\n{neg_df}')
    #     plt.scatter(neg_df[0], neg_df[1], label='neg')
    #     plt.scatter(pos_df[0], pos_df[1], label='pos')
    #
    #     # # plot the svm super_plane (or in this case, line)
    #     # weight = svm.coef_[0]
    #     # # get the intercept
    #     # bias = svm.intercept_[0]
    #     # k = -weight[0] / weight[1]
    #     # b = -bias / weight[1]
    #     # print(f'k is {k}, b is {b}')
    #     # xx=np.linspace(-1,1,100)
    #     # yy=k*xx+b
    #     # print(f'xx is {xx}')
    #     # print(f'yy is {yy}')
    #     # plt.plot(xx,yy)
    #
    #     w = svm.coef_[0]
    #     print(f'coef: {svm.coef_}')
    #     a = -w[0] / w[1]
    #     xx = np.linspace(-0.3, 0.7,10)
    #     yy = a * xx - (svm.intercept_[0]) / w[1]
    #
    #     # plot the parallels to the separating hyperplane that pass through the
    #     # support vectors
    #     b = svm.support_vectors_[0]
    #     yy_down = a * xx + (b[1] - a * b[0])
    #     b = svm.support_vectors_[-1]
    #     yy_up = a * xx + (b[1] - a * b[0])
    #
    #     # plot the line, the points, and the nearest vectors to the plane
    #     plt.plot(xx, yy, 'k-')
    #     plt.plot(xx, yy_down, 'k--')
    #     plt.plot(xx, yy_up, 'k--')
    #
    #     plt.legend()
    #     plt.show()


from scipy.interpolate import interp1d
from decimal import Decimal, ROUND_HALF_UP
def rec_alignment(rec_df: pd.DataFrame, rec_space:float=0.1):
    '''
    align the records in the same asc file
    Args:
        rec_df: the records to be aligned, columns are the amu value, rows are the simple record
    Returns:
        aligned record dataframe
    '''

    for idx, row in rec_df.iterrows():
        peaks, _ = find_peaks(row, height=1000, distance=10)
        if len(peaks) <= 0:
            return None
        # get the column name when peaks appear
        old_peaks = [float(rec_df.columns[peak]) for peak in peaks]
        try:
            if int(rec_df.columns[-1])-old_peaks[-1] > 0.5:
                old_peaks.append(int(rec_df.columns[-1]))
        except IndexError:
            print(f'columns:{rec_df.columns}')
            print(f'old_peaks:{old_peaks}')

        new_MS=[]
        amu_begin=float(rec_df.columns[0])
        if old_peaks[0]-amu_begin<0.5:
            last_old_peak=old_peaks[0]
            last_new_peak=int(Decimal(last_old_peak).quantize(Decimal('0'),rounding=ROUND_HALF_UP))
            old_peaks=old_peaks[1:]
        else:
            last_old_peak, last_new_peak= amu_begin,amu_begin

        for old_peak in old_peaks:
            if old_peak%1!=0:
                # new_peak=round(old_peak)
                new_peak=int(Decimal(old_peak).quantize(Decimal('0'),rounding=ROUND_HALF_UP))
            else:
                new_peak=old_peak
            if new_peak!=last_new_peak:
                # used to replace the np.arange because of rounding error
                # x_train=np.arange(last_old_peak,old_peak,rec_space)
                x_train=[x/10 for x in range(int(last_old_peak*10),int(old_peak*10+1))]
                y_train=row[x_train]
                x_train=np.linspace(0,1,len(x_train))
                interp_f=interp1d(x_train,y_train)
                x_test=np.linspace(0,1,int((new_peak-last_new_peak)/rec_space),endpoint=False)
                new_rec_frag=interp_f(x_test)
                new_MS.extend(new_rec_frag)
                last_old_peak=old_peak
                last_new_peak=new_peak
            else:
                raise ValueError('two peaks are too close')
        new_MS.append(row[rec_df.columns[-1]])
        new_MS=[int(x) for x in new_MS]
        rec_df.loc[idx]=new_MS
    return rec_df

def round_half(num:float):
    int_part=int(num)
    if num-int_part<0.25:
        return int_part
    elif num-int_part<0.75:
        return int_part+0.5
    else:
        return int_part+1

def rec_alignment_half(rec_df: pd.DataFrame, rec_space:float=0.1):
    '''
    re-align the records in the same asc file, the amu value is rounded to the nearest 0.5
    Args:
        rec_df: df that contain records from single asc file
    Returns:
        aligned dataframe
    '''

    for idx, row in rec_df.iterrows():
        peaks, _ = find_peaks(row, height=1000, distance=2)
        if len(peaks) <= 0:
            return None
        # get the column name when peaks appear
        old_peaks = [float(rec_df.columns[peak]) for peak in peaks]
        # old_peaks=sorted(list(set(old_peaks)-set(noise_peak)))
        try:
            if int(rec_df.columns[-1])-old_peaks[-1] > 0.25:
                old_peaks.append(int(rec_df.columns[-1]))
        except IndexError:
            print(f'columns:{rec_df.columns}')
            print(f'old_peaks:{old_peaks}')

        new_MS=[]
        amu_begin=float(rec_df.columns[0])
        if old_peaks[0]-amu_begin<0.2:
            last_old_peak=old_peaks[0]
            # last_new_peak=int(Decimal(last_old_peak).quantize(Decimal('0'),rounding=ROUND_HALF_UP))
            last_new_peak=round_half(float(last_old_peak))
            old_peaks=old_peaks[1:]
        else:
            last_old_peak, last_new_peak= amu_begin,amu_begin

        for old_peak in old_peaks:
            if old_peak%0.5!=0:
                # new_peak=round(old_peak)
                # new_peak=int(Decimal(old_peak).quantize(Decimal('0'),rounding=ROUND_HALF_UP))
                new_peak=round_half(float(old_peak))
            else:
                new_peak=old_peak
            if new_peak!=last_new_peak:
                # used to replace the np.arange because of rounding error
                # x_train=np.arange(last_old_peak,old_peak,rec_space)
                x_train=[x/10 for x in range(int(last_old_peak*10),int(old_peak*10+1))]
                y_train=row[x_train]
                x_train=np.linspace(0,1,len(x_train))
                interp_f=interp1d(x_train,y_train)
                x_test=np.linspace(0,1,int((new_peak-last_new_peak)/rec_space),endpoint=False)
                new_rec_frag=interp_f(x_test)
                new_rec_len=len(new_rec_frag)
                if new_rec_len!=int((new_peak-last_new_peak)/rec_space):
                    print(f'new rec len:{new_rec_len}')
                    print(f'new peak:{new_peak}')
                    print(f'last new peak:{last_new_peak}')
                    print(f'new peak-last new peak:{new_peak-last_new_peak}')
                    print(f'rec space:{rec_space}')
                    print(f'new rec len should be:{int((new_peak-last_new_peak)/rec_space)}')
                    raise ValueError('new rec len is not correct')
                new_MS.extend(new_rec_frag)
                last_old_peak=old_peak
                last_new_peak=new_peak
            else:
                continue
                # raise ValueError('two peaks are too close')
        new_MS.append(row[rec_df.columns[-1]])
        new_MS=[int(x) for x in new_MS]
        rec_df.loc[idx]=new_MS
    return rec_df

def get_peak_distribute(raw_data_df: pd.DataFrame) -> pd.Series:
    # create a empty df to store the peak position
    raw_peak_df = pd.DataFrame(columns=raw_data_df.columns)
    for idx, row in raw_data_df.iterrows(): # iterate each record in one asc file
        peak_idx, _ = find_peaks(row, distance=2, threshold=1000)
        raw_peak_df.loc[idx, raw_peak_df.columns[peak_idx]] = 1

    # get the peak distribution of all the peaks from an ASC file
    sum_peaks = raw_peak_df.sum(axis=0)
    return sum_peaks

def accumulate_peak_by_shift_window(amu_window_size: int,
                                    raw_peak_df: pd.DataFrame,
                                    sum_peaks: pd.Series) -> pd.Series:
    ls_sum = []
    shift = amu_window_size // 2 if amu_window_size % 2 == 1 else amu_window_size // 2 - 1
    amu_len = len(raw_peak_df.columns)
    for center_amu_idx in range(amu_len):
        window_left_idx = center_amu_idx - shift if center_amu_idx - shift >= 0 else 0
        window_right_idx = center_amu_idx + shift if center_amu_idx + shift < amu_len else amu_len - 1
        ls_sum.append(sum_peaks.iloc[window_left_idx:window_right_idx].sum())
    ls_sum = pd.Series(ls_sum, index=raw_peak_df.columns)
    return ls_sum

def get_raw_feat(threshold: int, peaks_apperance: pd.Series) -> set:
    peaks,_=find_peaks(peaks_apperance, distance=2, height=threshold)
    peaks=peaks_apperance.iloc[peaks].index
    peaks=[round_half(float(x)) for x in list(peaks)]
    return set(peaks)

def feature_extract(data_df: pd.DataFrame)->Collection:
    features=data_df.columns
    feature_df=data_df.loc[:,features]

    corr=feature_df.corr()
    feature_covid_corr=corr.loc['covid']
    feature_covid_corr.plot(kind='barh', figsize=(10,15))
    plt.grid()
    plt.title('Correlation between features and covid')
    plt.show()

    return feature_covid_corr[abs(feature_covid_corr)>0.4].index[:-1]

def calculate_vif(df):
    vif = pd.DataFrame()
    vif['index'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    return vif