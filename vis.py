import numpy as np
import constants
import pandas as pd
from matplotlib import pyplot as plt

# Define activity labels and colormap for plotting/visualization
ACTIVITY_LABELS = ['UNLABELED', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING',
                   'STAND_TO_SIT', 'SIT_TO_STAND', ' SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE',
                   'LIE_TO_STAND']
COLORMAP = ['whitesmoke', 'grey', 'lightcoral', 'lightsteelblue', 'khaki', 'peru', 'cyan', 'orange', 'g', 'royalblue',
            'plum', 'm', 'hotpink']


def _load_test_npz(path=constants.numpy_data_PATH):
    """

    Args:
        path: path to .npz file to load test data

    Returns: uncompressed numpy array

    """
    test_path = path + 'test.npz'
    test_arr = np.load(test_path, allow_pickle=True)['arr_0']

    return test_arr


def plot_test_data(eval_data_path=constants.results_PATH, results_path=constants.results_PATH):

    """ Load Evaluated data of test set from .csv (stores when you run evaluate.py)

    Args:
        eval_data_path: path to pre evaluated data
        results_path: path to save results
    """

    # Uncomment below f you want to load evaluated data from a numpy file
    # np_test = _load_test_npz()
    df_eval = pd.read_csv(eval_data_path + 'eval_data.csv')
    print(df_eval.head())
    df_test = df_eval
    df_test.columns = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', "Labels", "Predicted_Labels"]

    df_test["Labels"] = df_test["Labels"].astype(int)

    print(len(df_test))
    print(df_test.iloc[:, 6].value_counts())

    plot_dur = int(len(df_test) / constants.SLICE_OF_TEST_DATA_FOR_VIS)
    # converting row numbers into time duration
    time = [1 / float(constants.SAMPLING_FREQUENCY) * j for j in range(plot_dur)]

    # Define the figure and setting dimensions width and height
    fig = plt.figure(figsize=(20, 15))
    plt.suptitle("Signals, groundtruths and predictions on a slice of test data", fontsize=20)

    '''************************** Plot Accelerometer signals ********************************'''

    plt.subplot(5, 1, 1)
    # plot each signal component
    _ = plt.plot(time, df_test['Acc_X'].iloc[0:plot_dur], color='r', label='ACC_X')
    _ = plt.plot(time, df_test['Acc_Y'].iloc[0:plot_dur], color='g', label='ACC_Y')
    _ = plt.plot(time, df_test['Acc_Z'].iloc[0:plot_dur], color='b', label='ACC_Z')

    '''Plot and overlap groundtruth labels on Accelero signals '''

    pred = df_test["Labels"].iloc[0:plot_dur].to_list()
    color_list = list()
    for i in range(len(pred)):
        pred_index = pred[i]
        color = COLORMAP[pred_index]
        color_list.append(color)
    # print(color_list)
    plt.ylim([-5, 5])
    plt.vlines(time, -5, 5, linewidth=3, color=color_list)

    # y axis label
    figure_Ylabel = 'Acceleration'

    # Set the figure info defined earlier
    _ = plt.ylabel(figure_Ylabel)
    _ = plt.xlabel('Time in seconds (s)')
    _ = plt.title("Accelerometer signals")

    # location of the figure's legends
    _ = plt.legend(loc="upper right")  # upper left corner

    '''**************************************************************************************'''

    '''******************************* Plot Gyro signals ************************************'''

    plt.subplot(5, 1, 2)
    # plot each signal component
    _ = plt.plot(time, df_test['Gyro_X'].iloc[0:plot_dur], color='r', label='Gyro_X')
    _ = plt.plot(time, df_test['Gyro_Y'].iloc[0:plot_dur], color='g', label='Gyro_Y')
    _ = plt.plot(time, df_test['Gyro_Z'].iloc[0:plot_dur], color='b', label='Gyro_Z')

    '''Plot and overlap groundtruth labels on Gyro signals '''

    pred = df_test["Labels"].iloc[0:plot_dur].to_list()
    color_list = list()
    for i in range(len(pred)):
        pred_index = pred[i]
        color = COLORMAP[pred_index]
        color_list.append(color)
    # print(color_list)
    plt.ylim([-8, 8])
    plt.vlines(time, -8, 8, linewidth=3, color=color_list)

    # y axis label
    figure_Y1label = 'Angular Velocity [rad/s]'

    # Set the figure info defined earlier
    _ = plt.ylabel(figure_Y1label)
    _ = plt.xlabel('Time in seconds (s)')
    _ = plt.title("Gyro signals")

    # location of the figure's legends
    _ = plt.legend(loc="upper right")  # upper left corner

    '''**************************************************************************************'''

    '''***************************** Plot Groundtruth labels ********************************'''

    plt.subplot(5, 1, 3)
    # _ = plt.plot(time, df_test['Labels'],  label='Labels')
    plt.title('Groundtruth')
    pred = df_test["Labels"].iloc[0:plot_dur].to_list()

    color_list = list()
    for i in range(len(pred)):
        pred_index = pred[i]
        color = COLORMAP[pred_index]
        color_list.append(color)
    # print(color_list)
    plt.ylim([0, 5])
    plt.vlines(time, 0, 5, linewidth=3, color=color_list)

    '''******************************************************************************** *****'''

    '''***************************** Plot Predicted labels **********************************'''

    plt.subplot(5, 1, 4)
    # _ = plt.plot(time, df_test['Labels'],  label='Labels')
    plt.title('Predicted')
    pred = df_test["Predicted_Labels"].iloc[0:plot_dur].to_list()

    color_list = list()
    for i in range(len(pred)):
        pred_index = pred[i]
        color = COLORMAP[pred_index]
        color_list.append(color)
    # print(color_list)
    plt.ylim([0, 5])
    plt.vlines(time, 0, 5, linewidth=3, color=color_list)

    '''**************************************************************************************'''

    '''***************** Plot label color and Activity pair for reference *******************'''

    plt.subplot(5, 1, 5)

    plt.title('Colormap')
    plt.ylim([0, 1])
    plt.bar(range(13), 1, color=COLORMAP)
    plt.xticks(range(13), ACTIVITY_LABELS, rotation=75)

    '''**************************************************************************************'''

    plt.tight_layout()
    plt.savefig(results_path + 'test_data.png')


plot_test_data()
