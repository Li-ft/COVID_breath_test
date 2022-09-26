from sklearn import mixture
from config import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *


raw_data_df=asc_2df(r"C:\Users\maqly\Dropbox\NTA\Raffaele Correale - 19-mar-2021 ela Poli\mar 19 2021 4_1.ASC").transpose()

# print(data_df1.head(20))

# plot_rec_df(data_df, log_scale=True)
data_aligned_df=rec_alignment(raw_data_df.copy())

# print(data_df)
# data_df.to_csv(proj_path+'/data/aligned.csv')
#
most_corr_num=9
cols=most_corr_cols(data_aligned_df.transpose(), most_corr_num)
# data_df
# plot_rec_df(data_aligned_df, log_scale=True)
# plot_rec_df(data_aligned_df.loc[cols], log_scale=True)
#
plt.figure(figsize=(30,10))
plt.plot(data_aligned_df.loc[cols].mean(axis=0))
# plt.plot(raw_data_df.loc['127.453'], color='green', label='before')
# plt.plot(data_aligned_df.loc['127.453'], color='red', label='after')
plt.yscale('log')
x_ticks=range(int(data_aligned_df.columns[0]), int(data_aligned_df.columns[-1]), 1)
plt.xticks(x_ticks,x_ticks)
plt.grid()
plt.legend()
plt.title(f'average of {most_corr_num} most correlated records')
plt.savefig(fig_path+f'/average of {most_corr_num} most correlated records.png')
plt.show()