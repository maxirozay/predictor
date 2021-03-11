#%%
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflowjs as tfjs
import json

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

#columns = ['open', 'high', 'low', 'close', 'volume']

#def normalizeData(data):
#  df = pd.DataFrame(data, columns=column)
#  df_mean = df.mean()
#  df_std = df.std()
#  return (df - df_mean) / df_std

#with open("data.json", "r") as read_file:
#  data = json.load(read_file)
#  data = [[np.array([kline[1], kline[2], kline[3], kline[4], kline[5]]).astype(float) for kline in batch] for batch in data]
#  np.random.shuffle(data)
#  data = list(map(normalizeData, data))

data = []
for file in os.listdir("./data"):
  df = pd.read_csv(os.path.join("./data", file), usecols=[3,4,5,6], skiprows=1)
  df_mean = df.mean()
  df_std = df.std()
  data.append((df - df_mean) / df_std)

data = data[0:10]
n = len(data)
train_df = data[0:int(n*0.7)]
val_df = data[int(n*0.7):int(n*0.9)]
test_df = data[int(n*0.9):]
num_features = data[0].shape[1]
print(n, num_features)

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if self.label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(data[0].columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def plot(self, model=None, plot_col='open', max_subplots=3):
  inputs, labels = next(iter(self.test))
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
            label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

    plt.xlabel('Time')

WindowGenerator.plot = plot

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])
  return inputs, labels

WindowGenerator.split_window = split_window

def make_dataset(self, data):
  dsSum = None
  for ds in data:
    ds = np.array(ds, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=ds,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32)

    ds = ds.map(self.split_window)
    if dsSum:
      dsSum = dsSum.concatenate(ds)
    else:
      dsSum = ds
  return dsSum

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test

OUT_STEPS = 100
multi_window = WindowGenerator(input_width=500,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

MAX_EPOCHS = 5

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

multi_val_performance = {}
multi_performance = {}

multi_lstm_model = tf.keras.Sequential()
multi_lstm_model.add(tf.keras.layers.LSTM(32, return_sequences=False))
multi_lstm_model.add(tf.keras.layers.Dense(OUT_STEPS*num_features))
multi_lstm_model.add(tf.keras.layers.Reshape([OUT_STEPS, num_features]))

print('compile_and_fit')
history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

print('validation')
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
print('test')
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)

tfjs.converters.save_keras_model(multi_lstm_model, 'model')
print('done')
# %%
