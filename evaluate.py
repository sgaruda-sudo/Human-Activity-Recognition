import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras

'''User Defined Imports'''
import constants
from input_pipeline import dataset_building

activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT',
                   'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', 'STAND_TO_LIE', 'LIE_TO_STAND']

'''**************** All procedure to plot cm and classification report **********************'''


def _extract_data_from_prefetch_ds(ds_test, model):
    ############ Reproducing features,labels and predictions from prefetched dataset ################

    y_pred = []
    y_true = []
    y_feat = []
    for feats, labels in ds_test.take(-1):
        preds = model.predict(feats)

        np_preds = np.asarray(preds)

        np_pred_labels = np.argmax(np_preds, axis=2)
        y_pred.append(np.array(np_pred_labels).reshape(np_pred_labels.size, 1))

        label_t = np.argmax(tf.convert_to_tensor(labels), axis=2)
        y_true.append(np.array(label_t).reshape(label_t.size, 1))

        features_t = np.asarray(tf.convert_to_tensor(feats))
        y_feat.append(np.array(features_t).reshape(features_t.size, 1))

    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    np_y_feat = np.asarray(y_feat)

    return y_pred, y_true, np_y_feat


def _concatenate_and_reshape(y_pred, y_true, np_y_feat):
    """ Concatenate arrays then reshape(only for features) """

    for index in range(0, len(y_pred)):

        if (index == 0):
            y_pred_concat = y_pred[index]
            y_true_concat = y_true[index]
            y_feat_concat = np_y_feat[index]

        else:
            y_pred_concat = np.concatenate((y_pred_concat, y_pred[index]), axis=0)
            y_true_concat = np.concatenate((y_true_concat, y_true[index]), axis=0)
            y_feat_concat = np.concatenate((y_feat_concat, np_y_feat[index]), axis=0)

    y_feat_concat = np.reshape(y_feat_concat, (int(len(y_feat_concat) / 6), 6))
    # For cross verification purpose
    # print(len(y_pred_concat))
    # print(len(y_true_concat))
    # print(np.shape(y_feat_concat))

    # print(np.shape(y_pred),np.shape(y_true),y_pred)

    # sanity check for length of the arrays
    assert len(y_pred_concat) == len(y_true_concat) == len(y_feat_concat)

    return y_pred_concat, y_true_concat, y_feat_concat


def _save_eval_data_to_csv(y_pred_concat, y_true_concat, y_feat_concat, r_path=constants.results_PATH):
    """ ####################### extract features, groundtruth labels and predicted labels ##################### """

    """ Save the extracted features, groundtruth labels and predicted labels """

    df_eval = pd.DataFrame(y_feat_concat, columns=["Acc_X", "Acc_Y", "Acc_z", "Gyro_x", "Gyro_y", "Gyro_z"])
    df_eval["True_labels"] = y_true_concat
    df_eval["Predicted_labels"] = y_pred_concat

    df_eval.to_csv(r_path + 'eval_data.csv', index_label=False)


def _calc_nd_save_confusion_metric(y_true_concat, y_pred_concat, res_PATH=constants.results_PATH, SAVE_CM=True):
    """ Confusion matrix on true labels and predicted lables """

    # Confusion Matrix and plotting and saving
    confusion_mat = confusion_matrix(y_true_concat, y_pred_concat)
    print("\n Confusion Matrix: \n\n", confusion_mat)
    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    plt.suptitle("Human Acitivity Recognition Confusion matrix", fontsize=25)
    cm_plot = sns.heatmap(confusion_mat, xticklabels=activity_labels, yticklabels=activity_labels, annot=True, fmt="d")
    cm_fig = cm_plot.get_figure()
    if SAVE_CM:
        cm_fig.savefig(res_PATH + "confusion_matrix.png")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show();

    pass


def _classification_report_csv(report):
    dataframe = pd.DataFrame.from_dict(report)
    dataframe.to_csv(constants.results_PATH + 'classification_report.csv', index=False)


'''**************** Load the saved model and then compile it, before evaluation on test dataset ********************'''


def eval_model_cm_cr(model_path=constants.trained_model_PATH, results_PATH=constants.results_PATH, SAVE_RESULTS=True):
    """
    PURPOSE: To (evaluate the model and predict on test dataset). Also to calculate confusion matrix and
             classification report and if mentioned save the results.
    Args:
        SAVE_RESULTS: If True, save the results (confusion matrix and classification report)
        results_PATH: path to store results
        model_path: trained model path
    """
    # model = keras.models.load_model(constants.load_model_PATH+'20210201-1614/epochs:016-val_accuracy:0.710.h5')
    model = keras.models.load_model(model_path, compile=False)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
                  metrics=['accuracy'])

    # Print Model summary
    model.summary()

    # fetch test dataset for evaluation
    _, ds_test, _1 = dataset_building.window_and_build_ds()  # datasets.load()

    # Evaluate on test dataset
    test_history = model.evaluate(ds_test,
                                  batch_size=constants.N_BATCH_SIZE,
                                  verbose=1)

    # extract data from pre fetched dataset
    y_pred, y_true, np_y_feat = _extract_data_from_prefetch_ds(ds_test, model)

    # Concatenate and reshape data 
    y_pred_concat, y_true_concat, y_feat_concat = _concatenate_and_reshape(y_pred, y_true, np_y_feat)

    # save evaluated data to csv to visualize
    _save_eval_data_to_csv(y_pred_concat, y_true_concat, y_feat_concat)

    print("\n \nTest Loss:", np.mean(np.asarray(test_history[0])), "\tTest Accuracy :",
          np.mean(np.asarray(test_history[1])))

    # calculate confusion matrix and save it to .png in results folder
    _calc_nd_save_confusion_metric(y_true_concat, y_pred_concat, res_PATH=results_PATH, SAVE_CM=SAVE_RESULTS)

    # Calculate Classification Report and saving to csv
    if SAVE_RESULTS:
        cr = classification_report(y_true_concat, y_pred_concat, target_names=activity_labels, output_dict=True)
        _classification_report_csv(cr)

    cr = classification_report(y_true_concat, y_pred_concat, target_names=activity_labels)

    print("\n\nClassification Report :\n\n", cr, type(cr))

    ###############################################################################################

# eval_model_cm_cr(SAVE_RESULTS=True)
