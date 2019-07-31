import os
import random

import keras
import numpy as np
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.utils import plot_model
from matplotlib import pyplot as plt

from CustomScaler import CustomScaler
from simulation_list import SimulationList

simulations = SimulationList.jsonlines_load()

train_data = set([s for s in simulations.simlist])
new_data = [s for s in simulations.simlist if s.type != "original"]
random.seed(1)
test_data = set(random.sample(new_data, int(len(new_data) * 0.2)))
train_data -= test_data
print(len(train_data), len(test_data))
X = np.array(
    [[s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction] for s in
     train_data])
scaler = CustomScaler()
scaler.fit(X)
x = scaler.transform_data(X)
print(x.shape)
Y = np.array([s.water_retention_both for s in train_data])
print(Y.shape)
X_test = np.array(
    [[s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction] for s in
     test_data])
Y_test = np.array([s.mass_retention_both for s in test_data])
x_test = scaler.transform_data(X_test)
# print(X_test)
# print(X[0])
# exit()
random.seed()
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(random.randint(0, 100)), histogram_freq=1,
                                         batch_size=32, write_graph=True,
                                         write_grads=True, write_images=True, embeddings_freq=0,
                                         embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                         update_freq='epoch')

if os.path.exists("model.hd5"):
    model = load_model("model.hd5")
else:

    model = Sequential()
    model.add(Dense(6, input_dim=6, activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation="sigmoid"))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.summary()
    plot_model(model, "model.png", show_shapes=True, show_layer_names=True)
    model.fit(x, Y, epochs=200, callbacks=[tbCallBack], validation_data=(x_test, Y_test))
    loss = model.evaluate(x_test, Y_test)
    print(loss)
    if loss > 0.04:
        # exit()
        ...
    # print("-------------------------------------")
    # exit()
    model.save("model.hd5")

xrange = np.linspace(-0.5, 60.5, 100)
yrange = np.linspace(0.5, 5.5, 100)
xgrid, ygrid = np.meshgrid(xrange, yrange)
mcode = 1e23
wpcode = 15 / 100
wtcode = 15 / 100
gammacode = 0.7
testinput = np.array([[np.nan, np.nan, mcode, gammacode, wtcode, wpcode]] * 100 * 100)
testinput[::, 0] = xgrid.flatten()
testinput[::, 1] = ygrid.flatten()
testinput = scaler.transform_data(testinput)

print(testinput)
print(testinput.shape)
testoutput = model.predict(testinput)
outgrid = np.reshape(testoutput, (100, 100))

plt.pcolormesh(xgrid, ygrid, outgrid, cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.savefig("keras.png", transparent=True)
plt.show()