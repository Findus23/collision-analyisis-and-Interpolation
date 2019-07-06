import os
import random

import keras
import numpy as np
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from simulation_list import SimulationList

simulations = SimulationList.jsonlines_load()

random.shuffle(simulations.simlist)
train_data = [s for s in simulations.simlist]
test_data = [s for s in simulations.simlist if s.type != "original"]

X = np.array(
    [[s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction] for s in
     train_data])
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)
print(x.shape)
Y = np.array([s.water_retention_both for s in train_data])
print(Y.shape)
X_test = np.array(
    [[s.alpha, s.v, s.projectile_mass, s.gamma, s.target_water_fraction, s.projectile_water_fraction] for s in
     test_data])
Y_test = np.array([s.mass_retention_both for s in test_data])
x_test = scaler.transform(X_test)
# print(X_test)
# print(X[0])
# exit()
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                         write_grads=False, write_images=False, embeddings_freq=0,
                                         embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                         update_freq='epoch')

if os.path.exists("model.hd5") and False:
    model = load_model("model.hd5")
else:

    model = Sequential()
    model.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.summary()
    plot_model(model, "model.png", show_shapes=True, show_layer_names=True)
    model.fit(x, Y, epochs=400, callbacks=[tbCallBack], validation_split=0.02)
    loss = model.evaluate(x_test, Y_test)
    print(loss)
    # print("-------------------------------------")
    # exit()
    model.save("model.hd5")

xrange = np.linspace(-0.5, 60.5, 100)
yrange = np.linspace(0.5, 5.5, 100)
xgrid, ygrid = np.meshgrid(xrange, yrange)
mcode = 23e10
wpcode = 15 / 100
wtcode = 15 / 100
gammacode = 0.7
testinput = np.array([[np.nan, np.nan, mcode, gammacode, wtcode, wpcode]] * 100 * 100)
testinput[::, 0] = xgrid.flatten()
testinput[::, 1] = ygrid.flatten()
testinput = scaler.transform(testinput)

print(testinput)
print(testinput.shape)
testoutput = model.predict(testinput)
outgrid = np.reshape(testoutput, (100, 100))

plt.pcolormesh(xgrid, ygrid, outgrid, cmap="Blues", vmin=0, vmax=1)
plt.colorbar()
plt.savefig("keras.png", transparent=True)
plt.show()
