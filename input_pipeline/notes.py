import glob
import pandas as pd
import numpy as np
import os
import re
from constants import *
from pathlib import Path
from scipy.stats import zscore

data_dir = '../../HAR/HAPT DataSet/RawData/'

abs = r"C:/Users/Teja/Documents/_INFOTECH/sem5/DL_lab/dl-lab-2020-HAR-team14/HAR/HAPT DataSet/RawData/"

files = os.listdir(abs)

'''
if not os.path.exists(data_dir):
    raise ValueError
'''

list_file_paths = glob.glob(abs + "/*.txt")


def get_labels_df(file_apth):
    df_labels = pd.read_csv(file_apth,
                            header=None, delim_whitespace=True,
                            names=['exp_id', 'user_id', 'activity_id', 'StartTime', 'EndTime'], dtype=int)
    return df_labels


def load_file(filepath):
    # df = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return


# get labels dataframe
df_labels_data = get_labels_df(list_file_paths[122])

df_all_acc_data = pd.DataFrame(columns=['a_X', 'a_Y', 'a_Z'])
df_all_gyro_data = pd.DataFrame(columns=['g_X', 'g_Y', 'g_Z'])
# data frame for training
df_train = pd.DataFrame(columns=['a_X', 'a_Y', 'a_Z', 'g_X', 'g_Y', 'g_Z'])
# data frame for validation
df_valid = pd.DataFrame(columns=['a_X', 'a_Y', 'a_Z', 'g_X', 'g_Y', 'g_Z'])
# data frame for testing
df_test = pd.DataFrame(columns=['a_X', 'a_Y', 'a_Z', 'g_X', 'g_Y', 'g_Z'])
'''
exp = os.path.split(list_file_paths[5])[1]
exp = exp.split('.')[0]
sensor_type = exp.split('_')[0]
list_of_ids = re.findall("\d+",exp)
exp_id = int(list_of_ids[0])
user_id = int(list_of_ids[1])
'''

list_labels_train = []
list_labels_valid = []
list_labels_test = []

i = 0

for file_path in list_file_paths[0:2]:
    exp = os.path.split(file_path)[1]
    exp = exp.split('.')[0]
    sensor_type = exp.split('_')[0]

    if sensor_type == 'acc':
        # get ids mentioned in file name
        list_of_ids = re.findall("\d+", exp)
        # first decimal in list is experience ID
        exp_id = int(list_of_ids[0])
        # second decimal in list is user ID
        user_id = int(list_of_ids[1])

        acc_file = file_path
        gyro_file = acc_file.replace('acc', 'gyro')

        # get accel and gyro data
        df_acc_data = pd.read_csv(acc_file, names=['a_X', 'a_Y', 'a_Z'], delimiter=' ', )
        df_gyro_data = pd.read_csv(gyro_file, names=['g_X', 'g_Y', 'g_Z'], delimiter=' ', )

        assert len(df_acc_data) == len(df_gyro_data)
        df_all_acc_data = pd.concat([df_all_acc_data,
                                     df_acc_data.iloc[N_NOISY_SAMPLES:(-N_NOISY_SAMPLES), :]], axis=0)
        df_all_gyro_data = pd.concat([df_all_gyro_data,
                                      df_gyro_data.iloc[N_NOISY_SAMPLES:(-N_NOISY_SAMPLES), :]], axis=0)
        assert len(df_all_acc_data) == len(df_all_gyro_data)

        # df_exp = df_labels_data[df_labels_data['exp_id'] == int(exp_id)]
    # concatenate all raw data
    df_raw_all = pd.concat([df_all_acc_data, df_all_gyro_data], axis=1)
    print(len(df_raw_all))
    np_all_labels = np.zeros((len(df_raw_all), 1))
    #np_all_labels = np.zeros((len(df_raw_all), 3))
    # df_all_labels = pd.DataFrame.

    for idx, rows in df_labels_data.iterrows():
        if rows['exp_id'] == exp_id:
            # print('exp_id',exp_id)
            start, end = rows['StartTime'], rows['EndTime']
            label_values = int(rows['activity_id'])
            #label_usr_id = int(rows['user_id'])
            #label_exp_id = int(rows['exp_id'])
            # print(label_values)
            np_all_labels[start - N_NOISY_SAMPLES:end - N_NOISY_SAMPLES, 0] = label_values
            #np_all_labels[start - N_NOISY_SAMPLES:end - N_NOISY_SAMPLES, 1] = label_usr_id
            #np_all_labels[start - N_NOISY_SAMPLES:end - N_NOISY_SAMPLES, 2] = label_exp_id

    # users 1-21 - training data
    if user_id <= 21:
        list_labels_train.append(np_all_labels.tolist())
        df_train = pd.concat([df_train, df_raw_all], axis=0)

    # users 22-27 - testing data
    if 22 <= user_id <= 27:
        list_labels_test.append(np_all_labels.tolist())
        df_test = pd.concat([df_test, df_raw_all], axis=0)

    # users 28-30 - testing data
    if 28 <= user_id <= 30:
        list_labels_valid.append(np_all_labels.tolist())
        df_valid = pd.concat([df_test, df_raw_all], axis=0)

'''
df_train = df_train.apply(zscore).astype(float).round(decimals=5)
df_test = df_test.apply(zscore).astype(float).round(decimals=5)
df_valid = df_valid.apply(zscore).astype(float).round(decimals=5)

list_labels_train = np.concatenate(list_labels_train).astype('int32')
list_labels_test = np.concatenate(list_labels_test).astype('int32')
list_labels_valid = np.concatenate(list_labels_valid).astype('int32')
'''
print(len(train_data))
