# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Description: This program uses an artificial recurrent neural network 
# called Long Short Term Memory (LSTM) to predict the closing stock price of 
# a corporation using the past 60 day stock price.


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import date
# Identify the trading days
import datetime
import holidays

HOLIDAYS_US = holidays.US()
ONE_DAY = datetime.timedelta(days=1)

def business_day():
    next_day = datetime.date.today() + ONE_DAY
    pre_day = datetime.date.today() - ONE_DAY
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
        next_day = next_day + ONE_DAY
    while pre_day.weekday() in holidays.WEEKEND or pre_day in HOLIDAYS_US:
        pre_day = pre_day - ONE_DAY
    return next_day,pre_day
    
next_trading_day = business_day()[0].strftime("%Y-%m-%d")

    

enddate = date.today().strftime("%Y-%m-%d")

#Get the stock quote 
df = web.DataReader('^GSPC', data_source='yahoo', start='2011-01-01', 
                    end = enddate) 
df.shape
plt.figure(figsize=(16,8))
plt.title('Close Price History',fontsize = 24)
plt.plot(df['Close'])
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Close Price USD',fontsize = 18)
plt.show()

# Create a new data frame with only the closing price and convert it to an
# array.
data = df.filter(['Close'])

#Converting the dataframe to a numpy array
dataset = data.values

#Get /Compute the number of rows to train the model on, which is 80% of 

len(dataset)
training_data_len = math.ceil(len(dataset) *.90) 


# =============================================================================
# Now scale the data set to be values between 0 and 1 inclusive, I do this
#  because it is generally good practice to scale your data before giving it 
#  to the neural network.
# =============================================================================
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
# plt.plot(scaled_data)
# max(dataset)

# =============================================================================
# 
# Create a training data set that contains the past 60 day closing price 
# values that we want to use to predict the 61st closing price value.
# 
# =============================================================================
# Create the scaled training data set 
train_data = scaled_data[0:training_data_len, : ]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range (60, len(train_data)):
    x_train.append(train_data[i-60:i,0]) # python starts from 0, end at i -1
    y_train.append(train_data[i,0])
    
# =============================================================================
# append() function in python: list.append(item) add the item to the end of
# the list or add a column to the right size of the matrix
# =============================================================================

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train),np.array(y_train)
x_train.shape
y_train.shape

# Reshape the data into the shape accepted by the LSTM
x_train.shape[0]
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# =============================================================================
# Build the LSTM model to have two LSTM layers with 50 neurons and two 
# Dense layers, one with 25 neurons and the other with 1 neuron.
# =============================================================================
# Build the LSTM network model
model = Sequential()
model.add(LSTM(units = 50,return_sequences=True,
               input_shape=(x_train.shape[1],1)))
model.add(LSTM( units = 50, return_sequences = False))
model.add(Dense(units = 25))
model.add(Dense(units = 1))

# =============================================================================
# Compile the model using the mean squared error (MSE) loss function and
# the adam optimizer.
# =============================================================================
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# =============================================================================
# Train the model using the training data sets. Note, fit is another name 
# for train. Batch size is the total number of training examples present in 
# a single batch, and epoch is the number of iterations when an entire data 
# set is passed forward and backward through the neural network.
# =============================================================================
# Train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 10)

# Create a test data set.
test_data = scaled_data [training_data_len - 60 : , : ]
# Create the x test and y test
x_test = []
y_test = dataset[training_data_len : , : ]

# Get all of the rows from index 1603 to the rest and all of columns

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

# Convert x_test to a numpy array 
x_test = np.array(x_test)

# Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# Now get the predicted values from the model using the test data.
predictions_Neu = model.predict(x_test)

# Reserve Scaling
predictions = scaler.inverse_transform(predictions_Neu)

# =============================================================================
# Get the root mean squared error (RMSE), which is a good measure of how 
# accurate the model is. A value of 0 would indicate that the models predicted
#  values match the actual values from the test data set perfectly.
# =============================================================================

# rmse is a good matrics to evaluate the performance of the model
# Calculate the value of RMSE
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print("The Standard Error of the projection is " + str(rmse))

# Create a comparison chart
test_len = int(1.5*(len(dataset)-training_data_len))
train = data[(training_data_len-test_len):training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Create a chart
plt.figure (figsize = (16,8))
plt.title('S&P 500 Forecast v.s. Actual',fontsize=24)
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Close Price',fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Historical','Actual','Forecast'],fontsize=15 ,loc = 'lower right')
plt.show()

#Show the valid and predicted prices
# print(valid)

# Now use all of the historical data to pridicte the future One day price
new_df = df.filter(['Close'])

# Get last 60 days of data and convert list to vector
last_60_days = new_df[-60:].values

# Scale data to be [0 1]
last_60_days_scaled = scaler.transform(last_60_days)

Fcst = []
Fcst.append(last_60_days_scaled)
# Convert to np array
Fcst = np.array(Fcst)
Fcst = np.reshape(Fcst,(Fcst.shape[0],Fcst.shape[1],1))

# Generate predicted price 
Fcst_price = model.predict(Fcst)
Fcst_price = scaler.inverse_transform(Fcst_price)
Fcst_price = str(Fcst_price).strip('[]')
print("The Forecasted S&P 500 in " + next_trading_day + " is " + Fcst_price)




