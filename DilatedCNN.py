# -*- coding: utf-8 -*-
"""
@author: hannahannuniemi
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_load

# build the model
class ConvBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, dilation_rate, bias=False):
        super(ConvBlock, self).__init__()
        self.hidden_size = hidden_size,
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, dilation=dilation_rate)
        self.relu = nn.SELU()


    def forward(self, inputs):
        residual = inputs
        pads = (self.kernel_size[0] - 1) * self.dilation_rate
        inputs_padded = F.pad(inputs, (pads, 0))
        
        layer_out = self.relu(self.conv(inputs_padded))
        network_out = torch.add(residual, layer_out)
        return network_out

class DilatedConvNet(nn.Module):
    def __init__(self, n_layers, num_inputs, hidden_size, kernel_size, dilation_rate=1, bias=False):
        super(DilatedConvNet, self).__init__()
        self.n_layers = n_layers,
        self.hidden_size = hidden_size,
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.num_inputs = num_inputs
        
        # first layer
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=kernel_size, dilation=dilation_rate)
        self.relu1 = nn.SELU()
        self.skip_first = nn.Conv1d(1, 1, kernel_size=1, dilation=dilation_rate)
        # decrease the number of filters to hidden size
        self.conv2 = nn.Conv1d((hidden_size+1)*num_inputs, hidden_size, kernel_size=1, dilation=dilation_rate)
        
        # other layers
        other_layers = [ConvBlock(hidden_size=hidden_size, kernel_size=kernel_size, dilation_rate=2 ** i) for i in range(1, n_layers)]
        self.group = nn.Sequential(*other_layers)
        
        # last layer
        self.dense = nn.Conv1d(hidden_size, 1, kernel_size=1, dilation=dilation_rate)
        
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                

    def forward(self, inputs):
        # first layer
        # padding the input with a vector of zeros of the size of the receptive field
        pads = (self.kernel_size[0] - 1) * self.dilation_rate
        inputs_padded = F.pad(inputs, (pads, 0))
        
        input_layer = []
        for i in range(self.num_inputs):
          layer_out = self.relu1(self.conv1(inputs_padded[:,i,:].view(inputs_padded.shape[0],1,-1)))
          skip_out = self.skip_first(inputs[:,i,:].view(inputs.shape[0],1,-1))
          out = torch.cat((skip_out, layer_out), dim=1)
          input_layer.append(out)
        out = torch.cat(input_layer, dim=1)
        out = self.conv2(out)
        
        # other layers
        outputs = self.group(out)
        
        # last layer
        network_out = self.dense(outputs)
        outputs = network_out[:,:,-output_steps:] # output only last 48 hours
        return outputs


# function for training the model and monitoring training and validation error
# function modified from Pytorch tutorial: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#model-training-and-validation-code
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    since = time.time()
    val_loss_history = []
    train_loss_history = []

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                  outputs = model(inputs)
                  loss = criterion(outputs, targets)
                  # backward + optimize only if in training phase
                  if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            if (epoch % 10) == 0:
                print('Epoch {}/{} {} loss: {:.4f}'.format(epoch, num_epochs - 1, phase, epoch_loss))
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            
            scheduler.step(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, train_loss_history, val_loss_history

# function to split dataset to training and test sets
def split_dataset(data, split_point):
	train, test = data[0:split_point], data[split_point:data.shape[0]]
	return train, test

# function for converting training data into batches of equal length sequences of train_inputs and train_targets
# modified function, original from https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/
def split_equal_sequences(data, n_input, n_out, scaling=False):
  X, y = list(), list()
  in_start = 0
  # step over the training data one time step at a time
  for _ in range(len(data)):
    # define the end of the input sequence
    in_end = in_start + n_input
    out_end = in_end + n_out
    # ensure we have enough data for this instance
    if out_end < len(data):
      x_input = data[in_start:in_end, :]
      y_input = data[in_end:out_end, 0].reshape(n_out,1)
      # standardize data by removing the mean and scaling to unit variance
      if scaling:
        scaler = StandardScaler().fit(x_input)
        scaler2 = StandardScaler().fit(x_input[:,0].reshape(-1,1))
        x_input = scaler.transform(x_input)
        y_input = scaler2.transform(y_input.reshape(-1,1))
      
      X.append(x_input)
      y.append(y_input)
		# move along to create next batch
    in_start += 1
  return np.array(X), np.array(y)

# function for predicting next "output_steps" hours values based on "n_input" previous time steps using the trained model
def predict(model, input_data, n_input, device):
  # retrieve last observations for input data
  # last observations from input data have not been used in training
  inputs = input_data[-n_input:, :]
  # Standardize input data
  scaler = StandardScaler().fit(inputs)
  scaler2 = StandardScaler().fit(inputs[:,0].reshape(-1,1))
  inputs = scaler.transform(inputs)
  # reshape into [1, channels, n_input]
  inputs = inputs.reshape(1, inputs.shape[1], inputs.shape[0])
  inputs = torch.tensor(inputs, device=device, dtype=torch.float)
  # forecast the next "output_steps" hours
  model.eval()
  model.to(device)
  with torch.no_grad():
    inputs = inputs.to(device)
    outputs = model.forward(inputs)
    outputs = outputs.cpu().data.numpy()
    # scale back to original form
    outputs_rescaled = scaler2.inverse_transform(outputs.reshape(outputs.shape[2],outputs.shape[1]))
  return np.squeeze(outputs_rescaled)

# function for splitting test data to non-overlapping batches of "output_steps" hours
def split_test_data(data, n_out):
  y = list()
  in_start = 0
  for _ in range(len(data)):
    # define the end of the sequence
    in_end = in_start + n_out
    # check that we have enough data for this instance
    if in_end < len(data):
      batch = data[in_start:in_end]
      y.append(batch)
      in_start += n_out
  return np.array(y)

# function for evaluating predicted values
def evaluate_predictions(actual, predicted):
	scores = list()
	# calculate mean absolute error (MAE) score for each batch
	for i in range(actual.shape[0]):
		# calculate MAE
		mae = mean_absolute_error(actual[i, :], predicted[i, :])
		# store
		scores.append(mae)
	# calculate average MAE
	mae_average = sum(scores) / len(scores)
	return mae_average, scores

# function for evaluating model predictions using walk-forward validation
# test data is divided to batches of "output_steps" hours and for each batch MAE value is calculated
def evaluate_model(model, input_data, test_data, n_input, device):
	# walk-forward validation over each batch
  predictions = list()
  for i in range(len(test_data)):
    # predict the "output_steps" hours
    yhat = predict(model, input_data, n_input, device)
    # store the predictions
    predictions.append(yhat)
    # get real observation from the test data and add to input_data for predicting the next "output_step" hours
    input_data = np.concatenate([input_data, test_data[i, :, :]])
	# evaluate predictions for each batch
  predictions = np.array(predictions)
  mae_average, scores = evaluate_predictions(test_data[:,:,0], predictions)
  return mae_average, scores, predictions

# function for plotting predictions and true observations to same figure
# figure is created per test batch
# modified function, original from https://github.com/JEddy92/TimeSeries_Seq2Seq/blob/master/notebooks/TS_Seq2Seq_Conv_Full.ipynb
def plot_predictions(input_data, target_data, predictions, batch_ind, train_tail_len):
  
  input_series = input_data[:,0]
  pred_series = predictions[batch_ind,:]
  target_series = target_data[batch_ind,:,0]
    
  input_series_tail = np.concatenate((input_series[-train_tail_len:],target_series[:1]), axis=0) 
  x = input_series_tail.shape[0]
  pred_steps = pred_series.shape[0]
    
  plt.figure(figsize=(10,6))   
    
  plt.plot(range(1,x+1),input_series_tail)
  plt.plot(range(x,x+pred_steps),target_series,color='orange')
  plt.plot(range(x,x+pred_steps),pred_series,color='teal',linestyle='--')
    
  plt.title('Batch index %d' % batch_ind)
  plt.legend(['Input data','Target series','Predictions'])
  plt.xlabel('Hours')
  plt.ylabel('PM10 concentrations [micro g/m³]')


# function for naive forecast by taking last "output_steps" hours as the prediction for next "output_steps" hours and calculate MAE
def evaluate_naive(input_data, test, n_output):
  predictions = list()
  for i in range(len(test)):
    # take last "output_steps" hours from training data
    value = input_data[-n_output:, 0]
    predictions.append(value)
    # get real observation from test data and add to input_data
    input_data = np.concatenate((value, test[i, :, 0]), axis=0).reshape(-1,1)
	# evaluate predictions for each test batches
  predictions = np.array(predictions)
  mae_average, scores = evaluate_predictions(test[:,:,0], predictions)
  return mae_average, scores

##############################################################################

# Set the device
device = torch.device('cuda:0')
#device = torch.device("cpu")

# when using colab
from google.colab import drive
drive.mount('/content/drive')

# load data
df = pd.read_csv('/content/drive/My Drive/data/PM10_weather_data_LONG2.csv',  parse_dates=[['Year', 'Month', 'Day', 'Time']])
df.tail()

# print length of data and start and end time
data_size = df.shape[0]
print("Total amount of hours:", data_size)
first = df['Year_Month_Day_Time'].iloc[0]
last = df['Year_Month_Day_Time'].iloc[-1]
print("Data ranges from %s to %s." % (first, last))

# check min and max values for the concentration i.e. predictant
print(df['PM10 concentration (ug/m3)'].min())
print(df['PM10 concentration (ug/m3)'].max())

# check missing values
print("Missing values:\n")
print(df.isnull().sum(), "\n")
# put the time as index
df = df.set_index('Year_Month_Day_Time')

# use linear interpolation for filling the missing values
df_fill = df.interpolate()
print(df_fill.isnull().sum(), "\n")
print(df_fill.shape[0])

# plot PM10 concentration and amount of rain
plt.rcParams['figure.figsize']=(18,8)
fig1, ax1 = plt.subplots()
ax2 = ax1.twinx() 
ax1.bar(df_fill['Amount of rain (mm/h)'].index, df_fill['Amount of rain (mm/h)'], alpha=0.7, color='orange', label='Amount of rain')
ax2.plot(df_fill['PM10 concentration (ug/m3)'], label='PM10 concentration')
ax1.set_ylabel('Amount of rain (mm/h)')
ax2.set_ylabel('PM10 concentration (ug/m3)')
ax2.set_ylim(0, 250)
plt.gcf().autofmt_xdate()
plt.margins(x=0,y=0)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax2.set_title('PM10 concentration and amount of rain')

# plot PM10 concentration and temperature
plt.rcParams['figure.figsize']=(18,8)
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx() 
ax1.plot(df_fill['Temperature (degC)'], color='orange', label='Temperature')
ax2.plot(df_fill['PM10 concentration (ug/m3)'], label='PM10 concentration')
ax1.set_ylabel('Temperature (degC)')
ax2.set_ylabel('PM10 concentration (ug/m3)')
ax2.set_ylim(0, 250)
plt.gcf().autofmt_xdate()
plt.margins(x=0,y=0)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax2.set_title('PM10 concentration and temperature')
fig2.savefig('temp_plot.png')

# PM10 concentration and wind speed
plt.rcParams['figure.figsize']=(18,8)
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx() 
ax1.plot(df_fill['Wind speed (m/s)'], color='orange', label='Wind speed')
ax2.plot(df_fill['PM10 concentration (ug/m3)'], label='PM10 concentration')
ax1.set_ylabel('Wind speed (m/s)')
ax2.set_ylabel('PM10 concentration (ug/m3)')
ax2.set_ylim(0, 250)
plt.gcf().autofmt_xdate()
plt.margins(x=0,y=0)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax2.set_title('PM10 concentration and wind speed')

# plot histograms of input data
plt.rcParams['figure.figsize']=(18,6)
fig3, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.hist(df_fill['PM10 concentration (ug/m3)'], 50)
ax2.hist(df_fill['Amount of rain (mm/h)'], 10)
ax3.hist(df_fill['Temperature (degC)'], 40)
ax4.hist(df_fill['Wind speed (m/s)'], 40)
ax1.set_xlabel('PM10 concentration (ug/m3)')
ax2.set_xlabel('Amount of rain (mm/h)')
ax3.set_xlabel('Temperature (degC)')
ax4.set_xlabel('Wind speed (m/s)')
fig3.savefig('hist.png')

# check data types
df_fill.dtypes

# split input data to train and test batches
input_data = np.stack((df_fill['PM10 concentration (ug/m3)'].values, df_fill['Amount of rain (mm/h)'].values, df_fill['Temperature (degC)'].values, df_fill['Wind speed (m/s)'].values), axis=-1)
print(input_data.shape)
train, test = split_dataset(input_data, 18000)

# check the shapes of train and test sets
# and check that last value of train data and first value of test data are different
print(train.shape)
print(test.shape)
print(train[-1,0])
print(test[0,0])

# Length of input data (hours of history data) and lenght of prediction period
input_steps = 512 # 3 weeks
output_steps = 48

# split training data to input and target sequences. 
train_inputs, train_targets = split_equal_sequences(train, input_steps, output_steps, scaling=True)
print(train_targets.shape)
print(train_inputs.shape)

# reshape data to input format for convolution layer [batch, channels, steps] and tranform to tensors.
train_inputs = train_inputs.reshape(train_inputs.shape[0], train_inputs.shape[2], -1)
train_targets = train_targets.reshape(train_targets.shape[0], train_targets.shape[2], -1)
x = torch.tensor(train_inputs, device=device, dtype=torch.float)
y = torch.tensor(train_targets, device=device, dtype=torch.float)
print(x.size())
print(y.size())

# combine tensors to dataset
dataset = torch.utils.data.TensorDataset(x, y)

# divide the training dataset further to training and validation datasets.
idx = list(range(len(dataset)))
train_idx = idx[:15000]
val_idx = idx[15000:]
dataset_train = torch.utils.data.Subset(dataset, train_idx)
dataset_valid = torch.utils.data.Subset(dataset, val_idx)
print(len(dataset_train))
print(len(dataset_valid))

# load training and validation datasets to torch Dataloader format
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=15, shuffle=True, pin_memory=False)
validloader = torch.utils.data.DataLoader(dataset_valid, batch_size=5, shuffle=True, pin_memory=False)

# give parameters for the model and load to device
n_layers = 9
hidden_size = 3
kernel_size = 2
num_inputs = 4 # number of input features (number of input time series)
model = DilatedConvNet(n_layers=n_layers, num_inputs=num_inputs, hidden_size=hidden_size, kernel_size=kernel_size)
print(model.to(device))
model.to(device)

# using mean absolute error as a criterion
criterion = nn.L1Loss()
# using Adam and L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
# learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Feed a batch of data from the training data to test the network
with torch.no_grad():
    dataiter = iter(trainloader)
    values, labels = dataiter.next()
    values = values.to(device)
    print('Shape of the input tensor:', values.shape)

    y = model(values)
    print(y.shape)


# put training and validation sets to one dictionary for the use of training function
dataloaders_dict = {'train' : trainloader, 'val': validloader}

# train model and gather training and validation losses to list
model_trained, train_loss_history, val_loss_history = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=50)

# save model
model_filename = 'cnn_model.pth'
torch.save(model_trained.state_dict(), model_filename)

# plot training and validation error
plt.figure(figsize=(10,6))
plt.plot(np.arange(len(train_loss_history)), train_loss_history, label='train')
plt.plot(np.arange(len(val_loss_history)), val_loss_history, label='val')
plt.ylim([0,2])
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend()
plt.savefig('loss.png')

model_trained.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
model_trained.to(torch.device("cpu"))
print('Model loaded from %s' % model_filename)

# print weights of second layers convolution
model_trained.group[0].conv.weight

# split test data to batches of "output_steps" hours
test_batches = split_test_data(test, output_steps)
print(test_batches.shape)

# calculate MAE values for the model predictions
mae_average, mae_batches, predictions = evaluate_model(model_trained, train, test_batches, input_steps, device)
print(mae_average)
# MAE for the first 10 batches
mae_average_10batch = sum(mae_batches[:10]) / len(mae_batches[:10])
print(mae_average_10batch)

# plot the MAE over different test batches
plt.figure(figsize=(10,6))
plt.plot(range(len(mae_batches)), mae_batches)
plt.title('MAE over test batches')
plt.legend(['MAE'])
plt.xlabel('Batch number')
plt.ylabel('Mean absolute error')
plt.savefig('MAE_batches.png')

# plot predictions and true observations to same figure
plot_predictions(train, test_batches, predictions, batch_ind=0, train_tail_len=72)
plt.savefig('forecast0.png')

plot_predictions(train, test_batches, predictions, batch_ind=1, train_tail_len=72)
plt.savefig('forecast1.png')

plot_predictions(train, test_batches, predictions, batch_ind=2, train_tail_len=72)
plt.savefig('forecast2.png')

# calculate MAE values for the naive forecast
naive_mae_average, naive_mae_batches = evaluate_naive(train, test_batches, output_steps)
print(naive_mae_average)

