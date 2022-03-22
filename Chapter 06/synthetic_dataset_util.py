import tensorflow as tf
import numpy as np

baseline = 10
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

def get_time():
    return np.arange(4 * 365 + 1, dtype="float32")


def trend(time, slope=0):
    return slope * time

time = get_time()
series = trend(time, 0.1)  

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.sin(season_time * 2),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

split_time = 1000

def get_series():
    time  = get_time()
    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=42)
    return series




def get_x_train_x_valid():
    series = get_series()
    time = get_time()
    split_time = 1000
    return _get_x_train_x_valid(series, time, split_time)


def _get_x_train_x_valid(series,time, split_time):
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]
    return x_train,x_valid, time_train, time_valid

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset