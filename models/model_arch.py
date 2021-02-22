from tensorflow import keras
from tensorflow.keras import layers
import constants


# Model Architecture

def model_layers():

    
    inputs = layers.Input(constants.INPUT_SHAPE)

    out = layers.LSTM(units=128, return_sequences=True)(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.Dropout(0.25)(out)

    out = layers.LSTM(units=64, return_sequences=True)(out)
    out = layers.Dropout(0.10)(out)
    
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(12, activation="softmax")(out)

    model = keras.Model(inputs=inputs, outputs=out, name='HAR_model')

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
                  metrics=['accuracy'])
    


    #model.build(input_shape=(None, constants.INPUT_SHAPE[0], constants.INPUT_SHAPE[1]))
    return model
