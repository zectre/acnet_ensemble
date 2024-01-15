'''
Neural Network Model for Atmospheric correction in inland lakes in Tropical region
built on 11 May 2011
author: Oanh Thi La

'''
## 1. SETUP

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Activation, BatchNormalization, Dense, Dropout,Flatten, Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
import keras_tuner as kt
import tkinter
from tkinter.filedialog import askopenfilename


class ANNmodel():
    def load_trainingdata():
        TOA_xtrain = np.load('TOA_XTrain.npy')
        angles_xtrain = np.load('angles_XTrain.npy')
        AOT_xtrain = np.load('AOT_XTrain.npy')
        ytrain_iCOR = np.load('iCOR_YTrain.npy')

        TOA_xvali= np.load('TOA_XVali.npy')
        angles_xvali= np.load('angles_XVali.npy')
        AOT_xvali = np.load('AOT_XVali.npy')
        y_vali_iCOR = np.load('iCOR_Y_vali.npy')

        return TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR

    TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR = load_trainingdata()

    def build_model(hp):
        global hypermodel
        ## 3 Build a neural network model (Keras Functional API for Multiple inputs and mixed data)
        TOA_input = Input((3, 3, 8, 1))
        angles_input = Input((27,))
        AOT_input = Input((9,))

        # Create 3CN layers  ("valid" = without padding; "same" = with zero padding)
        x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='sigmoid',
                   kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(TOA_input)
        x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='sigmoid',
                   kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(x)
        x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='sigmoid',
                   kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(x)
        x = Flatten()(x)  # output shape =
        x = Model(inputs=TOA_input, outputs=x)
        x.summary()
        x = Reshape((288,))(x.output)
        # Combine x (TOA_extracted from CNN layers) and Angles data and AOT data
        combined = Concatenate(axis=1)([x, angles_input, AOT_input])

        ## Now apply Fully connected layers and then prediction on the combined data
        y = Dense(324, activation='sigmoid')(
            combined)  ##activation='relu' , kernel_initializer=initializer, bias_initializer=initializer
        y = Dropout(hp.Float("dropout1", 0, 0.5, step=0.1, default=0.5))(y)  # try to test from small dropout
        y = Dense(hp.Int("hidden1_size", 300, 400, step=10), activation='sigmoid')(y)  # activation='relu'
        y = Dropout(hp.Float("dropout2", 0, 0.5, step=0.1, default=0.5))(y)
        y = Dense(hp.Int("hidden2_size", 200, 300, step=10), activation='sigmoid')(y)  # activation='relu'
        y = Dropout(hp.Float("dropout3", 0, 0.5, step=0.1, default=0.5))(y)
        y = Dense(hp.Int("hidden3_size", 100, 200, step=10), activation='sigmoid')(y)
        y = Dropout(hp.Float("dropout4", 0, 0.5, step=0.1, default=0.5))(y)
        y = Dense(hp.Int("hidden4_size5", 20, 100, step=10), activation='sigmoid')(y)
        y = Dropout(hp.Float("dropout5", 0, 0.5, step=0.1, default=0.5))(y)
        y = Dense(5, activation='sigmoid')(y)

        ## output
        model = Model(inputs=[TOA_input, angles_input, AOT_input], outputs=y)
        model.summary()

        # Compiling model
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 0.9)), metrics=['mape', 'accuracy'])
        return model

    tuner = kt.Hyperband(build_model,
                         objective='val_accuracy',
                         max_epochs=50,
                         hyperband_iterations=2,
                         )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ## run the hyperparameter search. it be same as for model fit
    tuner.search([TOA_xtrain, angles_xtrain, AOT_xtrain], ytrain_iCOR,
                 validation_data=([TOA_xvali, angles_xvali, AOT_xvali], y_vali_iCOR),
                 callbacks=[early_stopping], batch_size=256, verbose=2)
    ## get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""The hyperparameter search is complete. 
    The optimal number of units in the 1st hidden layer is {best_hps.get('hidden1_size')}; 
    The optimal number of units in the 2nd hidden layer is {best_hps.get('hidden2_size')};
    The optimal number of units in the 3rd hidden layer is {best_hps.get('hidden3_size')};
    The optimal number of units in the 4th hidden layer is {best_hps.get('hidden4_size5')};
    The optimal 1st dropout is {best_hps.get('dropout1')};
    The optimal 2nd dropout is {best_hps.get('dropout2')};
    The optimal 3rd dropout is {best_hps.get('dropout3')};
    The optimal 4th dropout is {best_hps.get('dropout4')};
    The optimal 5th dropout is {best_hps.get('dropout5')};
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.""")

    ##TRAIN THE MODEL
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    train_history = model.fit([TOA_xtrain, angles_xtrain, AOT_xtrain], ytrain_iCOR,
                              validation_data=([TOA_xvali, angles_xvali, AOT_xvali], y_vali_iCOR),
                              batch_size=256, epochs=50, verbose=2)
    val_acc_per_epoch = train_history.history['val_mape']
    best_epoch = val_acc_per_epoch.index(min(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    # after finding out the best no of epochs. Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
    hypermodel = tuner.hypermodel.build(best_hps)
    # Retrain the model
    hypermodel.fit([TOA_xtrain, angles_xtrain, AOT_xtrain], ytrain_iCOR,
                   validation_data=([TOA_xvali, angles_xvali, AOT_xvali], y_vali_iCOR),
                   batch_size=256, epochs=best_epoch)  # epochs=best_epoch



