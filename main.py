### This example uses the historical prices of APPLE stock ###

# imports
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Dropout
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math


def create_model(input_lenght, output_lenght):

    model = Sequential()
    model.add(CuDNNLSTM(input_lenght, return_sequences=False,
              input_shape=[input_lenght, 1]))
    model.add(Dense(input_lenght * 2, Dropout(0.4)))
    model.add(Dense(input_lenght * 4, Dropout(0.3)))
    model.add(Dense(input_lenght * 2, Dropout(0.2)))
    model.add(Dense(input_lenght))
    model.add(Dense(output_lenght))
    model.compile(optimizer="sgd", loss="mean_squared_error")

    return model


def get_data():
    df = web.DataReader("AAPL", data_source="yahoo",
                        start="2012-01-01", end="2020-01-01")
    data = df.filter(["Close"])
    return data.values


def get_train_values(prev_values, next_values, scaled_data_arr):

    x_train = []
    y_train = []

    for i in range(prev_values, len(scaled_data_arr) - next_values):
        x_train.append(scaled_data_arr[i - prev_values: i])
        y_train.append(scaled_data_arr[i: i + next_values])
    # x_train, y_train = np.array(x_train), np.array(y_train)

    return x_train, y_train


def get_rmse(training_data_len, scaled_data, scaler, prev_values, next_values):

    test_data = scaled_data[training_data_len - prev_values:]

    scaled_data = np.reshape(scaled_data, (-1, 1))

    dataset = scaler.inverse_transform(scaled_data)

    # Create the datasets x_test and y_test
    x_test = []
    y_test = []
    for i in range(prev_values, len(test_data)):
        x_test.append(test_data[i - prev_values: i])
        y_test.append(dataset[i + len(test_data): i +
                      len(test_data) + next_values][0])

    # Get the models predicted price values
    predictions = model.predict(x_test)
    # Unscaling the values
    predictions = scaler.inverse_transform(predictions)

    ### Evaluate the model ###
    # Get the root mean squared error(RMSE)
    return np.sqrt(np.mean(((predictions - y_test) ** 2)))


# Constants #
PREVIOUS_VALUES = 365
NEXT_VALUES = 10
EPOCHS = 10
TRAINING_LEN = 0.8

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = get_data()

### Aca necesito un array -> [[val], [val]] ###
scaled_data = scaler.fit_transform(dataset)

# Create the scaled training data set
training_data_len = math.ceil(len(dataset) * TRAINING_LEN)

# Make the array one dimensional
scaled_data_arr = []
for elem in scaled_data:
    scaled_data_arr.append(elem[0])

train_data = scaled_data_arr[0:training_data_len]

x_train, y_train = get_train_values(
    PREVIOUS_VALUES, NEXT_VALUES, train_data)


model = create_model(PREVIOUS_VALUES, NEXT_VALUES)
# train the model
model.fit(x_train, y_train, epochs=EPOCHS)

# make a prediction
prediction = model.predict([x_train[0]])
# Unscale the data
prediction = scaler.inverse_transform(prediction)

print("last values: ")
print(str(dataset[PREVIOUS_VALUES: PREVIOUS_VALUES + NEXT_VALUES]))
#print("last value scaled: " + str(x_train[-1][-1]))
print("prediction:")
print(prediction)


print('Error (RMSE):')
print(get_rmse(training_data_len, scaled_data_arr,
               scaler, PREVIOUS_VALUES, NEXT_VALUES))
