import glob
import pandas as pd
import numpy as np
from natsort import natsorted
import re
from tqdm import trange
from scipy.stats import zscore
import tensorflow as tf

'''User Defined Imports'''
from constants import *

LABELS_COL_NAMES = ["experiment_number", "user_number", "activity_number", "start", "end"]
activity = ["NO ACTIVITY", "WALKING", "STANDING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "LAYING", "SITTING",
            "STAND_TO_LIE", "LIE_TO_SIT", "SIT_TO_LIE", "LIE_TO_STAND", "SIT_TO_STAND", "STAND_TO_SIT"]
ACC_COL = ['Acc_X', 'Acc_Y', 'Acc_Z']
GYRO_COL = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', "Labels"]
ALL_COLS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', "Labels"]


def _get_file_paths(path):
    """

    Args:
        path: path to raw data

    Returns: list of all files in the path

    """
    file_paths = natsorted(glob.glob(path + '*'))
    return file_paths


def _get_user_id_exp_id(filename: str) -> (int, int):
    """
    USE : to extract from experience id and user id from the string of file name
    :param filename: name of the file without the format extension
    :return:
    """
    # get ids mentioned in file name
    list_of_ids = re.findall("\d+", filename)

    # first decimal in list is experience ID
    # second decimal in list is user ID
    experience_id, user_id = int(list_of_ids[0]), int(list_of_ids[1])
    return experience_id, user_id


def _get_and_split_data(list_file_paths: list, df_all_labels):
    """

    Args:
        list_file_paths: list of all files containing user data from path in _get_file_paths
        df_all_labels: dataframe of all labels

    Returns: processed train,test and validation data(not tensorflow Dataset objects)

    """
    set_train = 0
    set_valid = 0
    set_test = 0

    for idx in trange(0, int((len(list_file_paths) - 1) / 2)):  # (0,1):#(int((len(list_file_paths) - 1) / 2)):

        f_name_w_format = os.path.split(list_file_paths[idx])[1]
        f_name = f_name_w_format.split('.')[0]
        exp_id, usr_id = _get_user_id_exp_id(filename=f_name)

        # get indices of respective exp and usr ids from labels dataframe
        indices = ((df_all_labels["experiment_number"] == exp_id) & (df_all_labels["user_number"] == usr_id))

        # extract i th experiment of j th user 's activity labels using the indices extracted above
        df_labels_n_exp_n_usr = df_all_labels.loc[indices]

        df_acc = pd.read_csv(list_file_paths[idx], sep=" ", names=ACC_COL)
        # df_acc['Labels'] = 0

        # replace acc with gyro in the file name read to read the gyro readings of same experience and same user
        gyro_f_path = list_file_paths[idx].replace('acc', 'gyro')
        df_gyro = pd.read_csv(gyro_f_path, sep=" ", names=GYRO_COL)
        df_gyro["Labels"] = 0

        # Checking if acc and gyro readings are of same length
        assert len(df_acc) == len(df_gyro)

        # concatenate gyro and accelerometer signals
        df_acc_gyro = pd.concat([df_acc, df_gyro], axis=1)

        # assign sequential labels to each time stamp
        for i in range(len(df_labels_n_exp_n_usr)):
            start_idx = df_labels_n_exp_n_usr["start"].iloc[i]
            end_idx = df_labels_n_exp_n_usr["end"].iloc[i]
            df_acc_gyro["Labels"].iloc[start_idx:end_idx] = df_labels_n_exp_n_usr["activity_number"].iloc[i]

        # Drop noise
        df_acc_gyro = df_acc_gyro[N_NOISY_SAMPLES:-N_NOISY_SAMPLES]

        # full dataframe
        if idx == 0:
            df_acc_gyro_all = df_acc_gyro
        else:
            if idx != 0:
                df_acc_gyro_all = pd.concat([df_acc_gyro_all, df_acc_gyro], axis=0)

        # append over all different user based on train,test,valid split
        # training users range
        if train_r_min <= usr_id <= train_r_max:
            if set_train == 0:
                df_train = df_acc_gyro
                set_train = 1
            else:
                df_train = pd.concat([df_train, df_acc_gyro], axis=0)
        # Test users range
        if test_r_min <= usr_id <= test_r_max:
            if set_test == 0:
                df_test = df_acc_gyro
                set_test = 1
            else:
                df_test = pd.concat([df_test, df_acc_gyro], axis=0)
        # validation users range
        if valid_r_min <= usr_id <= valid_r_max:
            if set_valid == 0:
                df_valid = df_acc_gyro
                set_valid = 1
            else:
                df_valid = pd.concat([df_valid, df_acc_gyro], axis=0)

        if usr_id == valid_r_max:
            # Sanity check if data has been distributed equally
            assert len(df_train) + len(df_valid) + len(df_test) == len(df_acc_gyro_all)

    return df_train, df_test, df_valid


def _save_data_in_npz(df_train_set, df_test_set, df_valid_set):
    """
    PURPOSE: to save train,test,validation data in numpy arrays
    Args:
        df_train_set: training dataframe
        df_test_set: testing dataframe
        df_valid_set: validation dataframe
    """
    np.savez('train', df_train_set.to_numpy())
    np.savez('test', df_test_set.to_numpy())
    np.savez('valid', df_valid_set.to_numpy())


def _apply_zscore(df_train_set, df_test_set, df_valid_set):
    """

    Args:
        df_train_set: training dataframe
        df_test_set: testing dataframe
        df_valid_set: validation dataframe

    Returns: Z scored data frame

    """
    z_df_train_set = df_train_set.iloc[:, :6].apply(zscore)
    z_df_test_set = df_test_set.iloc[:, :6].apply(zscore)
    z_df_valid_set = df_valid_set.iloc[:, :6].apply(zscore)

    z_df_train_set["Labels"] = df_train_set.iloc[:, 6]
    z_df_test_set["Labels"] = df_test_set.iloc[:, 6]
    z_df_valid_set["Labels"] = df_valid_set.iloc[:, 6]

    return z_df_train_set, z_df_test_set, z_df_valid_set


def _load_from_npz(path):
    train_path = path + 'train.npz'
    test_path = path + 'test.npz'
    valid_path = path + 'valid.npz'

    train_arr = np.load(train_path, allow_pickle=True)['arr_0']
    test_arr = np.load(test_path, allow_pickle=True)['arr_0']
    valid_arr = np.load(valid_path, allow_pickle=True)['arr_0']
    return train_arr, test_arr, valid_arr


# Function for one hot encoding with Number of classes =12
def _one_hot_enc(labels):
    """

    Args:
        labels: integer labels

    Returns: one hot encoded labels

    """
    one_hot_labels = []
    tmp = np.zeros((N_CLASSES,), dtype=int)
    for i in range(0, labels.size):
        tmp = np.zeros(N_CLASSES)
        x = labels[i]
        if 1 <= x <= N_CLASSES:
            tmp[x - 1] = 1
            one_hot_labels.append(tmp)
        elif x == 0:
            one_hot_labels.append(tmp)
    return one_hot_labels


def _split_np_data_nd_labels(np_data: np.ndarray):
    """

    Args:
        np_data: numpy array containing all features and labels

    Returns: numpy array of features, np array of categorical labels, np array of labels

    """
    np_main_data = np_data[:, :6]
    np_labels = (np_data[:, 6]).astype('int')
    np_categorical_labels = _one_hot_enc(np_labels)
    return np_main_data, np_categorical_labels, np_labels


# Window the Data and create a windowed dataset
def _create_window(feat, labels, win_size=N_WINDOW_SIZE, win_shift=N_WIN_SHIFT):
    """

    Args:
        feat: features
        labels: labels
        win_size: window size
        win_shift: window shifting width

    Returns:

    """
    list_feat = []
    list_labels = []
    for i in range(0, int(len(feat) / win_shift) - 1):
        list_feat.append(feat[i * win_shift:i * win_shift + win_size])
        list_labels.append(labels[i * win_shift:i * win_shift + win_size])

    return list_feat, list_labels


def _build_dataset(features, labels, type: str):
    """

    Args:
        features: np array of features
        labels: np array of labels
        type: dataset name

    Returns: prefetched tensorflow dataset

    """
    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    # if type == 'train':
    ds = ds.shuffle(N_SHUFFLE_BUFFER)
    ds = ds.cache()
    ds = ds.batch(N_BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def window_and_build_ds(f_path=numpy_data_PATH, raw_path=rawdata_PATH):
    """

    Args:
        f_path: path to numpy file of features and labels
        raw_path: raw data path

    Returns: prefetched tensorflow dataset

    """
    if SAVE_NP:
        list_file_paths = _get_file_paths(raw_path)

        # import labels file into a dataframe
        df_all_labels = pd.read_csv(list_file_paths[len(list_file_paths) - 1], sep=' ', names=LABELS_COL_NAMES)

        df_train_set, df_test_set, df_valid_set = _get_and_split_data(list_file_paths=list_file_paths,
                                                                      df_all_labels=df_all_labels)

        df_train_z, df_test_z, df_valid_z = _apply_zscore(df_train_set, df_test_set, df_valid_set)
        _save_data_in_npz(df_train_z, df_test_z, df_valid_z)

    # load data from saved numpy files
    np_train, np_test, np_valid = _load_from_npz(f_path)

    # split data into features and labels for respective set.
    np_train_data, np_train_labels, _ = _split_np_data_nd_labels(np_train)
    np_test_data, np_test_labels, _ = _split_np_data_nd_labels(np_test)
    np_valid_data, np_valid_labels, _ = _split_np_data_nd_labels(np_valid)

    # print(np.shape(np_train_labels), np_test_labels[0:5], np_valid_labels[0:5])

    # get windowed features and labels
    win_train_feat, win_train_labels = _create_window(np_train_data, np_train_labels)
    win_test_feat, win_test_labels = _create_window(np_test_data, np_test_labels, N_WINDOW_SIZE, N_WINDOW_SIZE)
    win_valid_feat, win_valid_labels = _create_window(np_valid_data, np_valid_labels, N_WINDOW_SIZE, N_WINDOW_SIZE)

    # build dataset for performance
    train_ds = _build_dataset(win_train_feat, win_train_labels, 'train')
    test_ds = _build_dataset(win_test_feat, win_test_labels, 'test')
    valid_ds = _build_dataset(win_valid_feat, win_valid_labels, 'valid')

    return train_ds, test_ds, valid_ds


# ds_train, ds_test, ds_valid = window_and_build_ds()

print("Hek")
