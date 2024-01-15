import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Activation, BatchNormalization, Dense, Dropout,Flatten, Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
import kerastuner as kt

## LOAD testing data (iCOR and insitu)
#source: \\140.116.80.130\home\AC-Net\InputforANN\NEWupdateDataset_May2021\dataforTESTING_update22May\Testing_insitulocation_VN
## or \\140.116.80.130\home\AC-Net\InputforANN\NEWupdateDataset_May2021\dataforTESTING_update22May\Testing_othercountries
def load_testingdata():
    TOA_xtesting = np.load('TOA_XVali.npy') #use val data bcs actly we dont need to test (only train)
    angles_xtesting = np.load('angles_XVali.npy')
    AOT_xtesting = np.load('AOT_XVali.npy')

    # y_test_iCOR_path = askopenfilename(title='Choose y_test iCOR files', filetypes=[("NPY", ".npy")])
    # ytest_iCOR = np.load(y_test_iCOR_path)
    # ytest_insitu_path = askopenfilename(title='Choose y_test insitu files', filetypes=[("NPY", ".npy")])
    # ytest_insitu = np.load(ytest_insitu_path)
    return TOA_xtesting, angles_xtesting, AOT_xtesting


TOA_xtesting, angles_xtesting, AOT_xtesting = load_testingdata()

# Load the trained model from the specified directory
hypermodel = tf.keras.models.load_model('my_trained_model.h5')

# Optionally, you can also load the tuner object and best hyperparameters
# loaded_tuner = kt.Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=30,
#     hyperband_iterations=2,
#     overwrite=True  # Set this to True if you want to overwrite the existing tuner
# )
# loaded_tuner = kt.tuners.tuner_utils.load_tuner(model_save_dir + 'my_tuner')


y_prediction = hypermodel.predict([TOA_xtesting, angles_xtesting, AOT_xtesting])  # prediction from x_test using model above
# path_save = tkinter.filedialog.asksaveasfilename(title=u'Save to excel file', filetypes=[("Excel", ".csv")])
path_save_csv = 'output.csv'
np.savetxt(path_save_csv, y_prediction, delimiter=",") #save to your specific folder
## or save y_prediction to npy to transfer to image in code "Result_Transfer.py"
from numpy import save
path_save_npy = 'output.npy'
# path_save = tkinter.filedialog.asksaveasfilename(title=u'Save npy file', filetypes=[("NPY", ".npy")])
save(path_save_npy, y_prediction)