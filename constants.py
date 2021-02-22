# ****************************************** Used in dataset_building.py *********************************************
# 250 samples = 5seconds at 50 hz frequency

N_NOISY_SAMPLES = 250

rawdata_PATH = 'HAPT_DataSet/RawData/'
#r"C:/Users/Teja/Documents/_INFOTECH/sem5/DL_lab/dl-lab-2020-HAR-team14/HAR/HAPT_DataSet/RawData/"

train_r_min, train_r_max = 1, 21
test_r_min, test_r_max = 22, 27
valid_r_min, valid_r_max = 28, 30

N_CLASSES = 12
# to save numpy file
SAVE_NP = 0

numpy_data_PATH =  'input_pipeline/'
#r"C:/Users/Teja/Documents/_INFOTECH/sem5/DL_lab/dl-lab-2020-HAR-team14/HAR/input_pipeline/"

logging_PATH = './log_dir/'
tensorboard_PATH = './log_dir/fit/'
checkpoint_PATH = './log_dir/cpts/'
csv_log_PATH = './log_dir/csv_log/'

# Hyper parameters
N_WINDOW_SIZE = 250
N_WIN_SHIFT   = 125

N_BATCH_SIZE = 128
N_SHUFFLE_BUFFER = 200

N_EPOCHS = 50
INPUT_SHAPE = (N_WINDOW_SIZE, 6)


# ********************************************************************************************************************
# evaluate.py
trained_model_PATH = 'weights/Accuracy_90.h5'
#'/home/malyalasa/New folder/HAR/log_dir/cpts/saved_model8_Acc82.83_Loss34/my_model'
#'log_dir/cpts/my_model-20210201T154258Z-001/my_model'

SAVE_CSV = 1
results_PATH = 'results/'

#**********************************************************************************************************************
# Vis.py
SAMPLING_FREQUENCY = 50

SLICE_OF_TEST_DATA_FOR_VIS = 20