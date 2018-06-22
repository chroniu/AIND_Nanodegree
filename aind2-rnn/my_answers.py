import numpy as np

import string

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

import keras


# transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    for i in range(0, len(series) - window_size, 1):
        X.append(series[i:i+window_size])
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

#  build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    #LSTM layer
    model.add(LSTM(input_shape=(window_size, 1), units=5))#window_size ))
    #output layer
    model.add(Dense(1))
    
    return model

### return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    lowercase_ascii = list(string.ascii_lowercase)
    replace = set(text) - set(punctuation + lowercase_ascii) 

    for x in replace:
        text = text.replace(x, ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
 
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size])
        outputs.append(text[i+window_size])
    return inputs,outputs

# build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    #LSTM layer
    model.add(LSTM(input_shape=(window_size, num_chars), units=200))
    #output layer
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    
    return model
