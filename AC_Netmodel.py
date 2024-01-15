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
import kerastuner as kt
import tkinter
from tkinter.filedialog import askopenfilename

## 2. LOAD DATA
def load_trainingdata(): #source: \\140.116.80.130\home\AC-Net\InputforANN\NEWupdateDataset_May2021\dataforTRAINING_update22May\2Train1Vali
    TOA_xtrain_path = askopenfilename(title='Choose TOA_xtrain files', filetypes=[("NPY", ".npy")])
    TOA_xtrain = np.load(TOA_xtrain_path)

    angles_xtrain_path = askopenfilename(title='Choose angles_xtrain files', filetypes=[("NPY", ".npy")])
    angles_xtrain = np.load(angles_xtrain_path)

    AOT_xtrain_path = askopenfilename(title='Choose AOT_xtrain files', filetypes=[("NPY", ".npy")])
    AOT_xtrain = np.load(AOT_xtrain_path)

    y_trainiCOR_path = askopenfilename(title='Choose iCOR y_train files', filetypes=[("NPY", ".npy")])
    ytrain_iCOR = np.load(y_trainiCOR_path)

    TOA_xvali_path = askopenfilename(title='Choose TOA_xvali files', filetypes=[("NPY", ".npy")])
    TOA_xvali = np.load(TOA_xvali_path)
    angles_xvali_path = askopenfilename(title='Choose angles_xvali files', filetypes=[("NPY", ".npy")])
    angles_xvali = np.load(angles_xvali_path)

    AOT_xvali_path = askopenfilename(title='Choose AOT_xvali files', filetypes=[("NPY", ".npy")])
    AOT_xvali = np.load(AOT_xvali_path)

    y_vali_iCOR_path = askopenfilename(title='Choose iCOR y_vali  files', filetypes=[("NPY", ".npy")])
    y_vali_iCOR = np.load(y_vali_iCOR_path)
    return TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR

TOA_xtrain, angles_xtrain, AOT_xtrain, ytrain_iCOR, TOA_xvali, angles_xvali, AOT_xvali, y_vali_iCOR = load_trainingdata()

## LOAD testing data (iCOR and insitu)
#source: \\140.116.80.130\home\AC-Net\InputforANN\NEWupdateDataset_May2021\dataforTESTING_update22May\Testing_insitulocation_VN
## or \\140.116.80.130\home\AC-Net\InputforANN\NEWupdateDataset_May2021\dataforTESTING_update22May\Testing_othercountries
def load_testingdata():
    TOA_xtesting_path = askopenfilename(title='Choose TOA_xtesting files', filetypes=[("NPY", ".npy")])
    TOA_xtesting = np.load(TOA_xtesting_path)
    angles_xtesting_path = askopenfilename(title='Choose angles_xtesting files', filetypes=[("NPY", ".npy")])
    angles_xtesting = np.load(angles_xtesting_path)

    AOT_xtesting_path = askopenfilename(title='Choose AOT_xtesting files', filetypes=[("NPY", ".npy")])
    AOT_xtesting = np.load(AOT_xtesting_path)

    # y_test_iCOR_path = askopenfilename(title='Choose y_test iCOR files', filetypes=[("NPY", ".npy")])
    # ytest_iCOR = np.load(y_test_iCOR_path)
    # ytest_insitu_path = askopenfilename(title='Choose y_test insitu files', filetypes=[("NPY", ".npy")])
    # ytest_insitu = np.load(ytest_insitu_path)
    return TOA_xtesting, angles_xtesting, AOT_xtesting


TOA_xtesting, angles_xtesting, AOT_xtesting = load_testingdata()


def build_model(hp):
    ## 3 Build a neural network model (Keras Functional API for Multiple inputs and mixed data)
    TOA_input = Input((3, 3, 8, 1))
    angles_input = Input((27,))
    AOT_input = Input((9,))

    # Create 3CN layers  ("valid" = without padding; "same" = with zero padding)
    x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='relu',
               kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(TOA_input)
    x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='relu',
               kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(x)
    x = Conv3D(filters=16, kernel_size=(1, 1, 3), strides=(1, 1, 1), padding='valid', activation='relu',
               kernel_regularizer=l2(0.01), activity_regularizer=l2(0.01))(x)
    x = Flatten()(x)  # output shape =
    x = Model(inputs=TOA_input, outputs=x)
    x.summary()
    x = Reshape((288,))(x.output)
    # Combine x (TOA_extracted from CNN layers) and Angles data and AOT data
    combined = Concatenate(axis=1)([x, angles_input, AOT_input])

    ## Now apply Fully connected layers and then prediction on the combined data
    y = Dense(324, activation='relu')(combined)  ##activation='relu' , kernel_initializer=initializer, bias_initializer=initializer
    y = Dropout(hp.Float("dropout1", 0, 0.5, step=0.1, default=0.5))(y)  # try to test from small dropout
    y = Dense(hp.Int("hidden1_size", 300, 400, step=10), activation='relu')(y)  # activation='relu'
    y = Dropout(hp.Float("dropout2", 0, 0.5, step=0.1, default=0.5))(y)
    y = Dense(hp.Int("hidden2_size", 200, 300, step=10), activation='relu')(y)  # activation='relu'
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
        hp.Float("learning_rate", 1e-4, 1e-2)), metrics=['mape', 'accuracy'])
    return model

    ## tune number of epochs

tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=30,
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
                          batch_size=256, epochs=30, verbose=2)
val_acc_per_epoch = train_history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# after finding out the best no of epochs. Re-instantiate the hypermodel and train it with the optimal number of epochs from above.
hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model
hypermodel.fit([TOA_xtrain, angles_xtrain, AOT_xtrain], ytrain_iCOR,
               validation_data=([TOA_xvali, angles_xvali, AOT_xvali], y_vali_iCOR),
               batch_size=256, epochs=best_epoch)


# def show_train_history(train_history, train, validation):
#     plt.plot(train_history.history[train])
#     plt.plot(train_history.history[validation])
#     plt.title('First phase')
#     plt.ylabel(train)
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()

# plt.figure(1)
# show_train_history(train_history, 'loss', 'val_loss')
# plt.figure(2)
# show_train_history(train_history, 'mape', 'val_mape')
# plt.figure(3)
# show_train_history(train_history, 'accuracy', 'val_accuracy')


# #  EVALUATE THE PREDICTION RESULT with insitu and iCOR
# scores1 = hypermodel.evaluate([TOA_xtesting, angles_xtesting, AOT_xtesting], ytest_insitu, verbose=0) # score = [loss, accuracy] as setup in the model.compile
# print(scores1)
# scores2 = hypermodel.evaluate([TOA_xtesting, angles_xtesting, AOT_xtesting], ytest_iCOR, verbose=0)  # score = [loss, accuracy] as setup in the model.compile
# print(scores2)

y_prediction = hypermodel.predict([TOA_xtesting, angles_xtesting, AOT_xtesting])  # prediction from x_test using model above
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to excel file', filetypes=[("Excel", ".csv")])
np.savetxt(path_save, y_prediction, delimiter=",") #save to your specific folder
## or save y_prediction to npy to transfer to image in code "Result_Transfer.py"
from numpy import save
path_save = tkinter.filedialog.asksaveasfilename(title=u'Save npy file', filetypes=[("NPY", ".npy")])
save(path_save, y_prediction)













# see weight of each layer
# CNNlayer1 = model.layers[1].get_weights()[0]
# CNNlayer2 = model.layers[2].get_weights()[0]
# CNNlayer3 = model.layers[3].get_weights()[0]
# dense_layer1 = model.layers[9].get_weights()[0]
# dense_layer2 = model.layers[11].get_weights()[0]
# dense_layer3 = model.layers[13].get_weights()[0]
# dense_layer4 = model.layers[15].get_weights()[0]
# dense_layer5 = model.layers[17].get_weights()[0]
# dense_layer6 = model.layers[19].get_weights()[0]



