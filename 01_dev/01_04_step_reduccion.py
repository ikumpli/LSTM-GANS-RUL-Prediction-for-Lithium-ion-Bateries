import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend as K
from itertools import chain

import warnings
warnings.filterwarnings('ignore')

path_origin, path_processed = "../data/original_data/", "../data/processed_data/"
models_path = "../data/processed_data/models/"

df_desc_final_pickle_load = pd.read_pickle(path_processed + 'df_desc_final.pkl')
df_desc_final_pickle = df_desc_final_pickle_load[df_desc_final_pickle_load.RUL >= 0 ]
df_desc_final_pickle["RUL"] = df_desc_final_pickle["RUL"].astype(float)

def x_and_y(df_desc_final_pickle, num_steps = 10, index = False):
    x, y, quantiles_final, new_values = [], [], [], []

    for i in range(1, 9):
        if index:
            batery_df = df_desc_final_pickle[df_desc_final_pickle.cell == i]
            quantiles = []
            for j in range(1, len(batery_df) - num_steps):
                x.append(np.array(batery_df.iloc[slice(j,num_steps+j), 1:].drop(columns=['RUL']).values))
                y.append(np.array([batery_df.iloc[slice(j,num_steps+j), 1:].RUL.values[-1], i]))
                quantiles.append(i)
        else:
            batery_df = df_desc_final_pickle[df_desc_final_pickle.index.get_level_values(0) == i]
            batery_df.index = pd.MultiIndex.from_tuples([(1, x) for x in range(1, batery_df.shape[0] + 1)])
            quantiles = []
            for j in range(1, len(batery_df) - num_steps):

                x.append(np.array(batery_df.loc[(1, slice(j,num_steps+j)), :].drop(columns=['RUL']).values))
                y.append(np.array([batery_df.loc[(1, slice(j,num_steps+j)), :].RUL.values[-1], i]))
                quantiles.append(i)

        quantiles_final.append(np.array_split(range(len(quantiles)), 4))
            
    x, y = np.array(x), np.array(y)

    for j in np.unique(y[:,1]):
        batery_column = y[:,1][y[:,1] == j]
        new_values.append([i+1 for i in range(4) for x in batery_column[quantiles_final[int(j-1)][i]].tolist()])
    y = np.append(y, np.array(list(chain(*new_values))).reshape(-1,1), axis = 1)
    y[:,0] = y[:,0].astype(float)
    return x, y

def customLoss(true,pred):
    diff = pred - true
    greater = K.greater(diff,0)
    greater = K.cast(greater, K.floatx()) 
    greater = greater + 1                
    return K.mean(K.square(diff))*greater

def model_run(steps, X_train, y_train):
    np.random.seed(26), tf.random.set_seed(26);
    early_stopping = keras.callbacks.EarlyStopping(patience=5000, restore_best_weights=True)
    checkpoint_first_model_Bi_LSTM = keras.callbacks.ModelCheckpoint(models_path + f'first_model_Bidirectional_LSTM_good_adapted_{steps+1}.h5', verbose = 0, save_best_only=True)

    model_lstm_bidirectional = keras.Sequential()
    forward_layer = keras.layers.LSTM(624, activation = keras.layers.LeakyReLU(alpha=0.3), recurrent_dropout = 0.05)
    backward_layer = keras.layers.LSTM(416, activation = tf.keras.layers.PReLU('zeros'), recurrent_dropout = 0.05, go_backwards = True)
    model_lstm_bidirectional.add(keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(X_train.shape[1], X_train.shape[2]) ))
    model_lstm_bidirectional.add(keras.layers.Dense(1, activation='relu'))

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model_lstm_bidirectional.compile(loss=customLoss, optimizer= optimizer, metrics=['mae'])
    historyBiLSTM = model_lstm_bidirectional.fit(X_train, y_train, epochs=20000, batch_size=16, validation_split=0.2, verbose=1, shuffle=False, callbacks=[checkpoint_first_model_Bi_LSTM, early_stopping])
    print(f"***************************************** MODEL OF {steps+1} STEPS HAS FINISHED ******************************************")


if __name__ == "__main__":
    model_list = [3,5,7,9]:
    for j in model_list:
        x, y = x_and_y(df_desc_final_pickle, num_steps = j-1, index = False)
        X_train,X_test,y_train_mod,y_test_mod = train_test_split(x, y,test_size=0.2, random_state=42, stratify=y[:,[1,2]])
        y_train, y_test = y_train_mod[:,0].reshape(-1,1), y_test_mod[:,0].reshape(-1,1)
        X_train, y_train, X_test, y_test = np.stack(X_train, axis = 0), np.stack(y_train, axis = 0), np.stack(X_test, axis = 0), np.stack(y_test, axis = 0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        model_run(j-1, X_train, y_train)

