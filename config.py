# need to be firstly configured if you want to use this script
proj_path=r'C:\Users\maqly\Desktop\thesis'
data_dir_path= r'C:\Users\maqly\Dropbox\NTA'
fig_path=proj_path+'/fig'

# used to plot different records
markers=['.','^','s','+','*']

# used in the data cleaning process
str_similarity_thresh=0.87

feature_self_corr_thresh=0.6
feature_covid_corr_thresh=0.2

most_corr_rec_num=9
mode='mean'


rec_space=0.1 # the difference between two adjacent records amu value
amu_window_size=5

# the range used in data acquisition
range_idx=4

amu_range = {
    0: ('10.0', '51.0'),
    1: ('10.0', '51.0'),
    2: ('49.0', '151.0'),
    3: ('149.0', '251.0'),
    4: ('249.0', '351.0')
}

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

