
# Research Overview
My research focuses on **predicting the ship trajectory**, which is a pivotal element in **reducing the risks of accidents or collisions** on water.


### Method Employed
In this project, I utilize the **LSTM (Long Short-Term Memory)** model. LSTMs are a type of recurrent neural network specialized in processing time-series data, capable of handling both short-term and long-term information concurrently.

### Dataset and Training
Based on the AIS data from Tokyo Bay, I trained the model using a dataset of 3.8 million entries. This model comprises approximately about 50,000 parameters. 

### As a result, the trajectory prediction accuracy has reached 97%.




# Project details
**Single step prediction step size**: 10

**Single-step loop prediction of long-term positions**: starting from the first position, the first 10 positions (true positions) predict the 11th position, and then the second position to the 11th position (predicted value) is a group, and the first 10 positions are predicted. 12 positions, this cycle predicts the value of a longer time, and its error will increase with the extension of time

**Multi-step prediction**: Assuming that the single-step prediction inputs 4 variables (lon, lat, cog, sog), the output is still 4 variables (lon, lat, cog, sog). If you want to directly predict two steps, you need to output 8 Variables {4 at the next moment + 4 at the next next moment}, namely (lon1, lat1, cog1, sog1, lon2, lat2, cog2, sog2)


## 1.Packages Import
Importing necessary libraries and packages for data manipulation, deep learning, and visualization.
```python
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger, ReduceLROnPlateau
from keras.optimizers import adam_v2
import transbigdata as tbd
import warnings
warnings.filterwarnings("ignore")
# Set seed parameters to facilitate reproduction
np.random.seed(120)
tf.random.set_seed(120)
```

### Data Reading
Loading the training and testing datasets from the train.csv and test.csv files, respectively.
```python
# Read data, updateDateFormat is positioning time, one point every two minutes, mmsi is track id
train = pd.read_csv("./train.csv",index_col=0)
test = pd.read_csv("./test.csv",index_col=0)
train.head()
```
![1693798712282](https://github.com/WenbinBAI/AI/assets/77138767/e01ccdd7-8a38-4421-a03b-0f9621359217)


### Feature Extraction and Label Data
create_dataset function helps in processing the dataset to create input sequences and corresponding labels for model training. It normalizes the data using provided maxmin (i.e., maximum and minimum values) for better model convergence.
```python
def create_dataset(data, window=10, max_min):
    """
    :param data:      Dataset of trajectories
    :param window:    How many data points per group
    :param max_min:   Used for normalization
    :return:          Data and labels
    """
    train_seq = []
    train_label = []
    m, n = max_min
    for traj_id in set(data['mmsi']):
        data_temp = data.loc[data.mmsi == traj_id]
        data_temp = np.array(data_temp.loc[:, ['lon', 'lat', 'sog', 'cog']])
        # Normalize
        data_temp = (data_temp - n) / (m - n)

        for i in range(data_temp.shape[0] - window):
            x = []
            for j in range(i, i + window):
                x.append(list(data_temp[j, :]))
            train_seq.append(x)
            train_label.append(data_temp[i + window, :])

    train_seq = np.array(train_seq, dtype='float64')
    train_label = np.array(train_label, dtype='float64')

    return train_seq, train_label
```

### Model Building
Defines the trainModel function that:
Builds a simple LSTM (Long Short Term Memory) neural network model for predicting the next location based on the input sequence.
Uses the Adam optimizer and Mean Squared Error (MSE) loss function.
Logs training progress and adjusts learning rate if validation accuracy plateaus.
Evaluates the model on the test set.
Saves the model.
```python
def trainModel(train_X, train_Y, test_X, test_Y):
    model = Sequential()
    model.add(LSTM(108, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
    # model.add(Dropout(0.3))
    model.add(Dense(train_Y.shape[1]))
    model.add(Activation("relu"))
    adam = adam_v2.Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['acc'])
    # Save changes in the loss function and accuracy during training
    log = CSVLogger(f"./log.csv", separator=",", append=True)
    # Used for automatic learning rate reduction
    reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1,
                               mode='auto', min_delta=0.001, cooldown=0, min_lr=0.001)
    # Train the model
    model.fit(train_X, train_Y, epochs=20, batch_size=32, verbose=1, validation_split=0.1,
                  callbacks=[log, reduce])
    # Evaluate with the test set
    loss, acc = model.evaluate(test_X, test_Y, verbose=1)
    print('Loss : {}, Accuracy: {}'.format(loss, acc * 100))
    # Save the model
    model.save(f"./model.h5")
    # Print neural network structure, count the number of parameters
    model.summary()
    return model
```

### Training
Preprocesses the training and testing data using create_dataset function and then trains the model using trainModel function.
```python
# Calculate normalization parameters
nor = np.array(train.loc[:, ['lon', 'lat', 'sog', 'cog']])
m = nor.max(axis=0)
n = nor.min(axis=0)
maxmin = [m, n]
# Window size
windows = 10
# Training set
train_seq, train_label = createSequence(train, windows, maxmin)
# Test set
test_seq, test_label = createSequence(test, windows, maxmin)
# Train the model. I trained it only 20 times. You can train it 100 times for higher accuracy.
model = trainModel(train_seq, train_label, test_seq, test_label)
# Load the pre-trained model
# model = load_model("./model.h5")
```



### Visualization
Plots changes in the model's accuracy and loss over training epochs.
```python
logs = pd.read_csv("./log.csv")

fig, ax = plt.subplots(2,2,figsize=(8,8))
ax[0][0].plot(logs['epoch'],logs['acc'], label='acc')
ax[0][0].set_title('acc')

ax[0][1].plot(logs['epoch'],logs['loss'], label='loss')
ax[0][1].set_title('loss')

ax[1][0].plot(logs['epoch'],logs['val_acc'], label='val_acc')
ax[1][0].set_title('val_acc')

ax[1][1].plot(logs['epoch'],logs['val_loss'], label='val_loss')
ax[1][1].set_title('val_loss')

plt.show()
```
![image](https://github.com/WenbinBAI/AI/assets/77138767/77546cdc-4a55-423e-851b-a97e6202282f)

# Prediction
Defines helper functions FNormalizeMult for reverse normalization (i.e., converting normalized data back to its original scale) and get_distance_hav for calculating distance between two geographical points.
```python
# Multi-dimensional De-normalization
def FNormalizeMult(y_pre, y_true, max_min):
    [m1, n1, s1, c1], [m2, n2, s2, c2] = max_min
    y_pre[:, 0] = y_pre[:, 0] * (m1 - m2) + m2
    y_pre[:, 1] = y_pre[:, 1] * (n1 - n2) + n2
    y_pre[:, 2] = y_pre[:, 2] * (s1 - s2) + s2
    y_pre[:, 3] = y_pre[:, 3] * (c1 - c2) + c2
    y_true[:, 0] = y_true[:, 0] * (m1 - m2) + m2
    y_true[:, 1] = y_true[:, 1] * (n1 - n2) + n2
    y_true[:, 2] = y_true[:, 2] * (s1 - s2) + s2
    y_true[:, 3] = y_true[:, 3] * (c1 - c2) + c2

    # Calculate the distance between the actual values and the predicted deviations
    y_pre = np.insert(y_pre, y_pre.shape[1],
                      get_distance_hav(y_true[:, 1], y_true[:, 0], y_pre[:, 1], y_pre[:, 0]), axis=1)

    return y_pre, y_true

# Haversine function
def hav(theta):
    s = np.sin(theta / 2)
    return s * s

# Calculate distance between coordinates in WGS84
def get_distance_hav(lat0, lng0, lat1, lng1):
    EARTH_RADIUS = 6371
    lat0 = np.radians(lat0)
    lat1 = np.radians(lat1)
    lng0 = np.radians(lng0)
    lng1 = np.radians(lng1)

    dlng = np.fabs(lng0 - lng1)
    dlat = np.fabs(lat0 - lat1)
    h = hav(dlat) + np.cos(lat0) * np.cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(h))
    return distance

```

### One_step prediction
Predicts the next trajectory point based on the current and past data points.
```python
test_points_ids = list(set(test['mmsi']))

for ids in test_points_ids[:1]:
    y_pre = []
    test_seq, test_label = createSequence(test.loc[test.mmsi == ids], windows, maxmin)

    y_true = test_label
    for i in range(len(test_seq)):
        y_hat = model.predict(test_seq[i].reshape(1, windows, 4))
        y_pre.append(y_hat[0])
    y_pre = np.array(y_pre, dtype='float64')
	# Denormalization
    f_y_pre, f_y_true = FNormalizeMult(y_pre, y_true, maxmin)

    print(f"Maximum: {max(f_y_pre[:, 4])}\nMinimum: {min(f_y_pre[:, 4])}\nMean: {np.mean(f_y_pre[:, 4])}\n"
          f"Variance: {np.var(f_y_pre[:, 4])}\nStandard Deviation: {np.std(f_y_pre[:, 4])}\nMedian: {np.median(f_y_pre[:, 4])}")

    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plt.plot(f_y_true[:, 0], f_y_true[:, 1], "ro", markersize=6, label='True Value')
    plt.plot(f_y_pre[:, 0], f_y_pre[:, 1], "bo", markersize=4, label='Predicted Value')
    bounds = [min(f_y_true[:, 0])-0.02, min(f_y_true[:, 1])-0.01, max(f_y_true[:, 0])+0.02, max(f_y_true[:, 1])+0.01]
    tbd.plot_map(plt, bounds, zoom=16, style=3)
    plt.legend(fontsize=14)
    plt.grid()
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)
    plt.title("MMSI:", fontsize=17)

    meanss = np.mean(f_y_pre[:, 4])
    plt.subplot(122)
    plt.bar(range(f_y_pre.shape[0]), f_y_pre[:, 4], label='Error')
    plt.plot([0, f_y_pre.shape[0]], [meanss, meanss], '--r', label="Mean")
    plt.title("Error Between Predicted and True Value", fontsize=17)
    plt.xlabel("Ship Trajectory Point", fontsize=14)
    plt.ylabel("Prediction Error (KM)", fontsize=14)
    plt.text(f_y_pre.shape[0]*1.01, meanss*0.96, round(meanss, 4), fontsize=14, color='r')
    plt.grid()
    plt.legend(fontsize=14)

    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.plot(f_y_pre[:, 2], "b-", label='Predicted Value')
    plt.plot(f_y_true[:, 2], "r-", label='True Value')
    plt.legend(fontsize=14)
    plt.title("Speed Prediction", fontsize=17)
    plt.xlabel("Ship Trajectory Point", fontsize=14)
    plt.ylabel("Speed/Knot", fontsize=14)
    plt.grid()

    plt.subplot(122)
    plt.plot(f_y_pre[:, 3], "b-", label='Predicted Value')
    plt.plot(f_y_true[:, 3], "r-", label='True Value')
    plt.legend(fontsize=14)
    plt.title("Direction Prediction", fontsize=17)
    plt.xlabel("Ship Trajectory Point", fontsize=14)
    plt.ylabel("Direction/Degree", fontsize=14)
    plt.grid()
```
![image](https://github.com/WenbinBAI/AI/assets/77138767/15671ef3-4412-4245-afb1-58c883d9480e)
![image](https://github.com/WenbinBAI/AI/assets/77138767/a7f856ce-937a-486e-8fc6-fd26a91e071e)


### Multi-step prediction, the ship position is at the last yellow point position
Predicts several future trajectory points iteratively by using predictions as input for subsequent steps.
```python
for ids in test_points_ids[:1]:
    test_seq, test_label = createSequence(test.loc[test.mmsi == ids], windows, maxmin)

    y_pre = []
    for i in range(len(test_seq)):
        y_hat = model.predict(test_seq[i].reshape(1, windows, 4))
        y_pre.append(y_hat[0])
    y_pre = np.array(y_pre, dtype='float64')
    
    # Get the true labels
    _, true_labels = FNormalizeMult(y_pre, np.copy(test_label), maxmin)
    
    # Start prediction from the fourth element
    for start_id in range(3, 4):
        # Single value prediction
        y_pre = []
        y_true = []
        pre_seq = test_seq[start_id]
        # Predict up to 15 minutes
        maxStep = min(15, test_seq.shape[0] - start_id)
        # Loop to make predictions
        for i in range(maxStep):
            y_hat = model.predict(pre_seq.reshape(1, windows, 4))
            y_pre.append(y_hat[0])
            y_true.append(test_label[start_id + i])
            # Get the next array
            pre_seq = np.insert(pre_seq, pre_seq.shape[0], y_pre[i], axis=0)[1:]

        y_pre = np.array(y_pre, dtype='float64')
        y_true = np.array(y_true, dtype='float64')
        f_y_pre, f_y_true = FNormalizeMult(y_pre, y_true, maxmin)

        plt.figure(figsize=(14, 6))
        plt.subplot(121)
        plt.plot(f_y_pre[:, 0], f_y_pre[:, 1], "bo", label='Predicted Value')
        plt.plot(f_y_true[:, 0], f_y_true[:, 1], "ro", label='True Value')
        plt.plot(true_labels[:start_id, 0], true_labels[:start_id, 1], "o", color='#eef200', label='Historical Position')
        bounds = [min(f_y_true[:, 0]) - 0.01, min(f_y_true[:, 1]) - 0.01, max(f_y_true[:, 0]) + 0.01, max(f_y_true[:, 1]) + 0.01]
        tbd.plot_map(plt, bounds, zoom=16, style=3)
        plt.legend(fontsize=15)
        plt.title(f'Number of Prediction Steps={maxStep}, Starting Position={start_id}', fontsize=17)
        plt.title(f'Real Trajectory vs Predicted Trajectory', fontsize=17)
        plt.xlabel("Longitude", fontsize=15)
        plt.ylabel("Latitude", fontsize=15)
        plt.grid()

        plt.subplot(122)
        plt.plot(np.arange(2, 2 * (maxStep) + 1, 2), f_y_pre[:, 4])
        plt.xticks(np.arange(2, 2 * (maxStep) + 1, 2))
        plt.title(f'Distance Error Over Time Iteration', fontsize=17)
        plt.xlabel("Time/Minutes", fontsize=15)
        plt.ylabel("Distance Error/Kilometers", fontsize=15)
        plt.grid()
        plt.show()
```
![image](https://github.com/WenbinBAI/AI/assets/77138767/cd5ff672-d62a-4210-bca9-758cd4b9d635)


### Predict 30-minute errors over the entire trajectory
```python
# Initialize a list to store all prediction errors
error_list = []

# Loop over the IDs (ships) in the test set (currently only the first one due to [:1])
for ids in test_points_ids[:1]:
    
    # Create a sequence for each ship's data 
    test_seq, test_label = createSequence(test.loc[test.mmsi == ids], windows, maxmin)
    
    # Set the prediction time to 30 (could be any desired prediction time)
    pre_time = 30
    
    # For each starting point in the test sequence, predict the next half of the pre_time
    for start_id in range(test_seq.shape[0]-int(pre_time/2)):
        
        # Lists to store single value predictions and their true values
        y_pre = []
        y_true = []
        
        # Get the initial sequence for prediction
        pre_seq = test_seq[start_id]
        
        # Loop for half the prediction time
        for i in range(int(pre_time/2)):
            
            # Make a prediction
            y_hat = model.predict(pre_seq.reshape(1, windows, 4))
            y_pre.append(y_hat[0])
            
            # Append the true value
            y_true.append(test_label[start_id+i])
            
            # Update the sequence for the next prediction (i.e., slide the window)
            pre_seq = np.insert(pre_seq, pre_seq.shape[0], y_pre[i], axis=0)[1:]
        
        # Convert lists to arrays for ease of computation
        y_pre = np.array(y_pre, dtype='float64')
        y_true = np.array(y_true, dtype='float64')
        
        # Normalize the predicted and true values
        f_y_pre, f_y_true = FNormalizeMult(y_pre, y_true, maxmin)
        
        # Append the errors in the predictions to the error list
        error_list.append(list(f_y_pre[:, 4]))

```
