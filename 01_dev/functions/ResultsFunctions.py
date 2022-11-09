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

###################################
#### GETTING THE NECESARY DATA ####
###################################
path_origin, path_processed = "../02_data/original_data/", "../02_data/processed_data/"
models_path = "../03_models/models/"
palette = ['#264653','#2A9D8F','#85CAC2','#DFF6F4' ,'#E9C46A','#F4A261','#E76F51','#C53D1B', '#7E2711']
cells = [f'\Cell{x}\*' for x in range(1, 9)]
font = {'size': 16, 'color': 'black', 'weight': 'bold'}

path_origin, path_processed = "../02_data/original_data/", "../02_data/processed_data/"
models_path = "../03_models/models/"
palette = ['#264653','#2A9D8F','#85CAC2','#DFF6F4' ,'#E9C46A','#F4A261','#E76F51','#C53D1B', '#7E2711']
cells = [f'\Cell{x}\*' for x in range(1, 9)]
font = {'size': 16, 'color': 'black', 'weight': 'bold'}

df_desc_final_pickle_load = pd.read_pickle(path_processed + 'df_desc_final.pkl')
df_desc_final_pickle = df_desc_final_pickle_load[df_desc_final_pickle_load.RUL >= 0 ]
df_desc_final_pickle["RUL"] = df_desc_final_pickle["RUL"].astype(float)
df_desc_final_pickle_not_index = df_desc_final_pickle.reset_index(col_level=0)

##############################################################
#### FIRST WE WILL TRAIN THE MODEL WITH THE WHOLE DATASET ####
##############################################################
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
    print(x.shape, y.shape)
    return x, y


def stratified_split(x, y, stratify_2 = True):

    if stratify_2: X_train,X_test,y_train_mod,y_test_mod = train_test_split(x, y,test_size=0.2, random_state=42, stratify=y[:,[1,2]])
    else: X_train,X_test,y_train_mod,y_test_mod = train_test_split(x, y,test_size=0.2, random_state=42, stratify=y[:,[1]])

    y_train, y_test = y_train_mod[:,0].reshape(-1,1), y_test_mod[:,0].reshape(-1,1)

    X_train = np.stack(X_train, axis = 0)
    y_train = np.stack(y_train, axis = 0)

    X_test = np.stack(X_test, axis = 0)
    y_test = np.stack(y_test, axis = 0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    return X_train, X_test, y_train, y_test, y_train_mod, y_test_mod


##########################################
#### LOSS FUNCTIONS AND MODEL LOADING ####
##########################################
def customLoss(true,pred):
    diff = pred - true
    greater = K.greater(diff,0)
    greater = K.cast(greater, K.floatx()) 
    greater = greater + 1                
    return K.mean(K.square(diff))*greater

def customLoss_numpy(true,pred):
    diff = pred - true
    greater = np.max(diff)
    greater = greater + 1     
    return np.mean(greater*(diff**2))

def return_metric(model_name, X_train, X_test, y_train, y_test, y_train_mod, y_test_mod):
    model = keras.models.load_model(models_path + f'{model_name}.h5', custom_objects={'customLoss': customLoss})
    ypred_train = model.predict(X_train)
    print('Train -->', r2_score(y_train, ypred_train), mean_absolute_error(y_train, ypred_train), np.sqrt(mean_squared_error(y_train, ypred_train)), customLoss_numpy(ypred_train, y_train))
    ypred = model.predict(X_test)
    print('Test -->', r2_score(y_test, ypred), mean_absolute_error(y_test, ypred), np.sqrt(mean_squared_error(y_test, ypred)), customLoss_numpy(ypred, y_test))
    print('')
    return ypred_train, ypred


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

########################################
#### HERE STARTS THE PLOT FUNCTIONS ####
########################################
def scatter_cycle_plot(df_desc_final_pickle_load, battery, ypred, y_test, y_test_mod, index = False):

    differences = (y_test - ypred).flatten()
    list_soh = []
    for k, v in enumerate(differences):
        if y_test_mod[:,1][k] == battery:
            df_modify = df_desc_final_pickle_load.loc[y_test_mod[:,1][k],:]
            yellow = df_modify.head(1).RUL[0] - df_modify[df_modify['SoH_max_ch']>=82].tail(1).index.get_level_values(0)[0]
            red = df_modify.RUL.min() if len(df_modify[df_modify['SoH_max_ch']<70].head(1)) == 0 else df_modify[df_modify['SoH_max_ch']<70].head(1).RUL.values[0]
            list_soh.append(v)

    plt.scatter(range(len(list_soh)), np.sort(list_soh), s = 110, color='k')
    plt.xticks(fontsize=22, family='serif'), plt.yticks(fontsize=22, family='serif');
    plt.grid(linestyle='--',linewidth=1.5, alpha = 0.4);
    plt.xlabel('Instances', fontsize=24, family='serif'), plt.ylabel('Cycle error', fontsize=24, family='serif'), plt.title(f'Prediction cell {battery}', fontsize=24, fontweight='bold', family='serif');

    yellow_lim = np.max(list_soh) + 50 if np.max(list_soh) > yellow else 100
    red_lim = np.min(list_soh) - 100 if np.min(list_soh) < 50 else 100
    annotation = abs((yellow + yellow_lim) - (red+red_lim)) / 40

    plt.axhspan(red, red+red_lim, facecolor='#f50400', alpha=0.3), plt.axhspan(yellow + yellow_lim, yellow, facecolor='#f4b41a', alpha=0.4), \
                                                                   plt.axhspan(yellow, 0, facecolor='#68da3e', alpha=0.4), plt.axhspan(0, red, facecolor='#f4b41a', alpha=0.4);
    for k,v in enumerate(np.sort(list_soh)): 
        if int(np.round(v)) < 0: plt.annotate( str(int(np.round(v))), (k, v + annotation), fontsize=18, color = 'r', fontweight='bold', rotation=90)
        else: plt.annotate( str(int(np.round(v))), (k, v + annotation), fontsize=18, color = 'k', fontweight='bold',rotation=90);
    plt.ylim(red+red_lim, yellow + yellow_lim);


def scatter_plot_prediction(ypred, y_test, y_test_mod):
    fig = plt.figure(figsize=(50, 20))
    for x in range(1, 9):
        plt.subplot(2, 4, x)
        scatter_cycle_plot(df_desc_final_pickle_load, x, ypred, y_test, y_test_mod, False)

def line_plot_result(ypred, y_test, title):
    ypred_ordered = ypred[np.argsort(y_test, axis=0)]
    y_test_ordered = np.sort(y_test, axis=0)

    plt.plot(y_test_ordered,  label='Real', color='#E76F51', linewidth=3);
    plt.plot(ypred_ordered.reshape(-1,1), 'bo',label='Prediction', color='#264653', linewidth=1);
    plt.xticks(fontsize=16, family='serif'), plt.yticks(fontsize=16, family='serif'), plt.grid(linestyle='--',linewidth=1.5, alpha = 0.5);
    plt.xlabel('Instances', fontsize=16, family='serif'), plt.ylabel('Cycles', fontsize=16, family='serif'), plt.title('Model results' + title, fontsize=16, fontweight='bold', family='serif');
    plt.legend(fontsize=16);
    for text in plt.gca().get_legend().get_texts():
        plt.setp(text, family='serif')

def bar_plot_comparison(nombres, list_red, list_yellow, list_green):
    X_axis = np.arange(len(nombres))
    fig1, ax1 = plt.subplots(figsize=(20, 5))
    plt.bar(X_axis - 0.2, list_red, 0.2, color = '#f50400', label = 'Dangerous', alpha = 0.3)
    plt.bar(X_axis, list_yellow, 0.2, color = '#f4b41a', label = 'Aceptable', alpha = 0.3)
    plt.bar(X_axis + 0.2, list_green, 0.2, color = '#68da3e', label = 'Excelent', alpha = 0.3)
    plt.xticks(X_axis, nombres, fontsize = 15, family='serif'), plt.yticks(fontsize = 15, family='serif');
    plt.xlabel("Models", color = 'k',fontsize=15, family='serif'), plt.ylabel("% of bateries", color = 'k',  fontsize=15, family='serif')
    plt.title("Model results", color = 'k', weight = 'bold', fontsize=18, family='serif')
    plt.ylim(0, 100)
    for p in ax1.patches: plt.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2, p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=14, weight = 'bold', family='serif')

def green_yellow_red(ypred, y_test, y_test_mod):
    differences = (y_test - ypred).flatten()
    red_list, yellow_list, green_list = [], [], []
    for k, v in enumerate(differences):
        df_modify = df_desc_final_pickle_load[df_desc_final_pickle_load.index.get_level_values(0) == y_test_mod[k][1]]
        yellow = df_modify.head(1).RUL.values[0] - df_modify[df_modify['SoH_max_ch']>=82].tail(1).index.get_level_values(1)[0]
        red = df_modify.RUL.min() if len(df_modify[df_modify['SoH_max_ch']<70].head(1)) == 0 else df_modify[df_modify['SoH_max_ch']<70].head(1).RUL.values[0]
        if (v <= yellow) & (v >= 0): green_list.append(k)
        if (v <= red): red_list.append(k)
        if ((v > red) & (v < 0)) | (v > yellow) : yellow_list.append(k)
    sum_list = len(red_list) + len(yellow_list) + len(green_list)
    return 100*(len(green_list)/sum_list), 100*len(yellow_list)/sum_list, 100*len(red_list)/sum_list