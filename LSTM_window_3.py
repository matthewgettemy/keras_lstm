"""
Working through tutorial on implementing an LSTM network in Keras.
Tutorial found here: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
"""

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('international-airline-passengers.csv',
                            usecols=[1],
                            engine='python',
                            skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# now the data should be distributed between 0 and 1
# this is because LSTMs are sensitive to the scale of the input data
# specifically when using the sigmoid or tanh activation functions
# because they saturate very quickly when deviating from the origin
# plt.plot(dataset)
# plt.show()

# use cross validation to expose a trained model to unseen data...
# you can run your model against the training data to see how it does,
# but a better test is to show it unseen data. When we are looking at
# time series data the sequence of points is very important, so we would
# not want to randomly sample our training/validation datasets, we will
# need to break the time series data into chunks

# split into train and test sets
train_size = int(len(dataset) * 0.67)  # 2/3
test_size = len(dataset) - train_size  # 1/3
train = dataset[0:train_size, :]
test = dataset[train_size:len(dataset), :]


def create_dataset(dataset, look_back=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        data_X.append(a)
        data_Y.append(dataset[i + look_back, 0])
    return numpy.array(data_X), numpy.array(data_Y)


# using the create_dataset offset function, create a predictor for
# both the training and validation data sets.
look_back = 3
train_X, train_Y = create_dataset(train, look_back)
test_X, test_Y = create_dataset(test, look_back)

# LSTM expects our data to be in the form: [samples, time steps, features]
# and right now our data is in the form [samples, features]
train_X = numpy.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = numpy.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

"""
The Network:
    - a visible layer with 1 input
    - a hidden layer with 4 LSTM blocks
    - an output layer with a single value prediction

Notes:
    - default sigmoid activation function is used for LSTM blocks
    - trained for 100 epochs
    - batch size = 1
"""
# fit the model
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

# now we can estimate the performance of the model using the train and test data
# make predictions
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# invert predictions to using original scaler
train_predict = scaler.inverse_transform(train_predict)
train_Y = scaler.inverse_transform([train_Y])
test_predict = scaler.inverse_transform(test_predict)
test_Y = scaler.inverse_transform([test_Y])

# calculate the overall root mean squared error
train_score = math.sqrt(mean_squared_error(train_Y[0], train_predict[:, 0]))
print('Train score: {} RMSE'.format(train_score))
test_score = math.sqrt(mean_squared_error(test_Y[0], test_predict[:, 0]))
print('Test score: {} RMSE'.format(test_score))

# shift train predictions for plotting
train_predict_plot = numpy.empty_like(dataset)
train_predict_plot[:, :] = numpy.nan
train_predict_plot[look_back: len(train_predict) + look_back, :] = train_predict

# shift predictions for plotting
test_predict_plot = numpy.empty_like(dataset)
test_predict_plot[:, :] = numpy.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict

# plot baseline data and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()
