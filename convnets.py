import os
from tensorflow.python import keras
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data_dir = 'C:\\Users\\Mirela\\PycharmProjects\\untitled2\\data_dir'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

from matplotlib import pyplot as plt

temp = float_data[:, 1]

plt.plot(range(len(temp)), temp)
plt.show()

plt.plot(range(1440), temp[:1440])
plt.show()
# data normalization
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
float_data = normalizer.fit_transform(float_data)

look_back = 720
sampling_rate = 6

train_generator = keras.preprocessing.sequence.TimeseriesGenerator(float_data, float_data, length=look_back,
                                                                   sampling_rate=sampling_rate, stride=1, start_index=0,
                                                                   end_index=200000, shuffle=True,
                                                                   batch_size=128)

val_generator = keras.preprocessing.sequence.TimeseriesGenerator(float_data, float_data, length=look_back,
                                                                 sampling_rate=sampling_rate, stride=1,
                                                                 start_index=200001, end_index=300000,
                                                                 batch_size=128)

test_gen = keras.preprocessing.sequence.TimeseriesGenerator(float_data, float_data, length=look_back,
                                                            sampling_rate=sampling_rate, stride=1,
                                                            start_index=300000, end_index=None,
                                                            batch_size=128)

val_steps = (300000 - 200001 - look_back)
test_steps = (len(float_data) - 300001 - look_back)

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(look_back // sampling_rate,
                                      float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_generator,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_generator,
                              validation_steps=val_steps)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
