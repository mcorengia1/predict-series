### This example uses the historical prices of APPLE stock ###

# imports
from keras.models import Sequential
from keras.layers import Dense
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler

####### Get Data #######
df = web.DataReader("AAPL", data_source="yahoo", start="2012-01-01", end="2020-01-01")
data = df.filter(["Close"])
dataset = data.values

####### Pre procesar data #######
### Aca necesito un array -> [[val], [val]] ###
PREVIOUS_VALUES = 200
NEXT_VALUES = 10

# Scale data, must be two dimensional
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Make the array one dimensional
scaled_data_arr = []
for elem in scaled_data:
    scaled_data_arr.append(elem[0])

# Use this only if im gonna use part of the data as testing data
# training_data_len = math.ceil(len(scaled_data_arr) * 0.8)
# train_data = scaled_data_arr[:training_data_len]

x_train = []
y_train = []
for i in range(PREVIOUS_VALUES, len(scaled_data_arr) - NEXT_VALUES):
    x_train.append(scaled_data_arr[i - PREVIOUS_VALUES : i])
    y_train.append(scaled_data_arr[i : i + NEXT_VALUES])

# x_train, y_train = np.array(x_train), np.array(y_train)


####### AI #######
# model definition
model = Sequential()
model.add(Dense(PREVIOUS_VALUES, input_shape=[PREVIOUS_VALUES]))
model.add(Dense(PREVIOUS_VALUES * 3))
model.add(Dense(PREVIOUS_VALUES * 2))
model.add(Dense(PREVIOUS_VALUES))
model.add(Dense(NEXT_VALUES))
model.compile(optimizer="sgd", loss="mean_squared_error")

# train the model
model.fit(x_train, y_train, epochs=10)

# make a prediction
print("prediction!!!")
prediction = model.predict([x_train[0]])

print(prediction)
# Unscale the data
prediction = scaler.inverse_transform(prediction)

print("last value: " + str(dataset[-1]))
print("last value scaled: " + str(x_train[-1][-1]))
print("prediction:")
print(prediction)