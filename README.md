
# Research Overview
My research focuses on **predicting the ship trajectory**, which is a pivotal element in **reducing the risks of accidents or collisions** on water.


### Method Employed
In this project, I utilize the **LSTM (Long Short-Term Memory)** model. LSTMs are a type of recurrent neural network specialized in processing time-series data, capable of handling both short-term and long-term information concurrently.

### Dataset and Training
Based on the AIS data from Tokyo Bay, I trained the model using a dataset of 3.8 million entries. This model comprises approximately about 50,000 parameters. As a result, the trajectory prediction accuracy has reached 97%.

### Tools and Platforms
For this research, I executed my programs on Google's Colab platform. 





# Project details
**Single step prediction step size**: 10

**Single-step loop prediction of long-term positions**: starting from the first position, the first 10 positions (true positions) predict the 11th position, and then the second position to the 11th position (predicted value) is a group, and the first 10 positions are predicted. 12 positions, this cycle predicts the value of a longer time, and its error will increase with the extension of time

**Multi-step prediction**: Assuming that the single-step prediction inputs 4 variables (lon, lat, cog, sog), the output is still 4 variables (lon, lat, cog, sog). If you want to directly predict two steps, you need to output 8 Variables {4 at the next moment + 4 at the next next moment}, namely (lon1, lat1, cog1, sog1, lon2, lat2, cog2, sog2)

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
# 设置种子参数，方便复现
np.random.seed(120)
tf.random.set_seed(120)
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


