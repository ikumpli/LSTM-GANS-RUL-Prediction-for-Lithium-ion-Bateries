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
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from keras.regularizers import l2

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Bidirectional, LSTM, Reshape, RepeatVector, TimeDistributed
from keras.layers import BatchNormalization, Activation
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam


def line_plot_result(ypred, y_test, title):
    ypred_ordered = ypred[np.argsort(y_test, axis=0)]
    y_test_ordered = np.sort(y_test, axis=0)
    yellow_bottom = (y_test_ordered - 600)
    yellow_top = (y_test_ordered + 400)
    X_axis = np.arange(len(y_test_ordered))

    plt.plot(y_test_ordered,  label='Real', color='#2A9D8F', linewidth=3);
    plt.plot(yellow_bottom,  linewidth=1, linestyle='--', color='green');
    plt.plot(yellow_top,  linewidth=1, linestyle='--', color='#f50400');

    plt.plot(X_axis, [-300] * len(y_test_ordered),  linewidth=2, linestyle='--', color='#E9C46A');
    plt.plot(ypred_ordered.reshape(-1,1), 'bo', label='Prediction', color='#264653', linewidth=1);
    
    plt.fill_between(X_axis, y_test_ordered.flatten(),yellow_bottom.flatten(), color='#68da3e', alpha=.4)
    plt.fill_between(X_axis, yellow_bottom.flatten(), [-300] * len(y_test_ordered), color='#f4b41a', alpha=.4)
    plt.fill_between(X_axis, yellow_top.flatten(), y_test_ordered.flatten(), color='#f4b41a', alpha=.4)
    plt.fill_between(X_axis, [4000] * len(y_test_ordered), yellow_top.flatten(), color='#f50400', alpha=.3)

    plt.xticks(fontsize=16, family='serif'), plt.yticks(fontsize=16, family='serif'), plt.grid(linestyle='--',linewidth=1.5, alpha = 0.3);
    plt.xlabel('Instances', fontsize=16, family='serif'), plt.ylabel('Cycles', fontsize=16, family='serif'), plt.title('Model results' + title, fontsize=16, fontweight='bold', family='serif');
    plt.legend(loc='upper right', fontsize=16, bbox_to_anchor=(1.45,1), borderaxespad=0);
    plt.ylim(-300, 4000)

    for text in plt.gca().get_legend().get_texts(): plt.setp(text, family='serif')