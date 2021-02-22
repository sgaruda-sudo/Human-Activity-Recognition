from absl import app, flags
import os
import datetime
import tensorflow as tf

'''User Defined Imports'''
import constants
import evaluate
from input_pipeline import dataset_building
from models import model_arch

# Set the 'train' flag to True to run the script the run in training mode
FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')


def main(argv):
    if FLAGS.train:

        # setup pipeline- prefetched tensorflow dataset
        ds_train, ds_test, ds_val = dataset_building.window_and_build_ds()  # datasets.load()

        # tensor board call back
        log_dir = constants.tensorboard_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callbk = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                                                            update_freq='epoch')

        os.makedirs(constants.checkpoint_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M"))
        cpt_folder = constants.checkpoint_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        checkpoint_dir = cpt_folder + "/epochs:{epoch:03d}-val_accuracy:{val_accuracy:.3f}.h5"

        # check point to save the model based on improving validation accuracy
        checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                               verbose=1,
                                                               save_best_only=False,
                                                               mode='max', save_weights_only=False,
                                                               save_freq='epoch')
        # csv logger call back
        log_file_name = constants.csv_log_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
        csv_callbk = tf.keras.callbacks.CSVLogger(log_file_name, separator=',', append=True)

        # list of call backs
        callbacks_list = [tensorboard_callbk, checkpoint_callbk, csv_callbk]

        # get the model
        full_model = model_arch.model_layers()
        # print model summary
        full_model.summary()
        # train the model, change the hyper parameters in constants.py
        history = full_model.fit(ds_train, epochs=constants.N_EPOCHS, validation_data=ds_val, verbose=1,
                                 callbacks=callbacks_list)
        # Evaluate the model
        test_history = full_model.evaluate(ds_test,
                                           batch_size=constants.N_BATCH_SIZE,
                                           verbose=1)
    else:
        # Evalulate and calculate confusion matrix and classification report and save them
        evaluate.eval_model_cm_cr(model_path=constants.trained_model_PATH, results_PATH=constants.results_PATH,
                                  SAVE_RESULTS=True)


if __name__ == "__main__":
    app.run(main)
