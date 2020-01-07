from __future__ import absolute_import, division, print_function

import pathlib 

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 

import time 
import os 

start = time.time()
var_testing = "Ozone"
col_names = ["Ozone", "Cell Temp", "Cell Pressure", "Flow Rate", "PDV"]
dataset = pd.read_csv("O3_scope_20180808123554.csv", names = col_names)
# dataset.tail()

train_dataset = dataset.sample(frac = 0.90, random_state = 1)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(train_dataset[["Wind Speed", "Wind Direction", "Temp", "AirU"]], diag_kind = "kde")
# plt.show()

train_stats = train_dataset.describe()
train_stats.pop(var_testing)
train_stats = train_stats.transpose()

train_labels = train_dataset.pop(var_testing)
test_labels = test_dataset.pop(var_testing)

def norm(x):
	return (x - train_stats["mean"]) / train_stats["std"]
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
	model = keras.Sequential([
		layers.Dense(32, activation = tf.nn.relu, input_shape = [len(train_dataset.keys())]),
		# layers.Dropout(0.5),
		layers.Dense(32, activation = tf.nn.relu),
		# layers.Dropout(0.5),
		layers.Dense(32, activation = tf.nn.sigmoid),
		# layers.Dropout(0.5),
		layers.Dense(1)
	])
	
	optimizer = tf.keras.optimizers.RMSprop(0.001)

	model.compile(loss = "mean_squared_error",
			  optimizer = optimizer,
			  metrics = ["mean_absolute_error", "mean_squared_error"])
	return model 

model = build_model()

model.summary()

example_batch = normed_train_data
example_result = model.predict(example_batch)

class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 50 == 0: 
			os.system("clear")
			for j in range(4):
				b = "Working" + "."*j
				if j == 3:
					b = b + str(epoch)
				print(b, end = "\r")
				time.sleep(0.05)
		
EPOCHS = 1000

# history = model.fit(normed_train_data,
# 					train_labels,
# 					epochs = EPOCHS,
# 					validation_split = 0.2,
# 					verbose = 0,
# 					callbacks = [PrintDot()]
# 					)

# hist = pd.DataFrame(history.history)
# hist["epoch"] = history.epoch 
# hist.tail()

def plot_history(history):
	hist = pd.DataFrame(history.history)
	hist["epoch"] = history.epoch 
	
	plt.figure()
	plt.xlabel("Epoch")
	plt.ylabel("Mean Abs Error [Resistance]")
	plt.plot(hist["epoch"], hist["mean_absolute_error"],
			label = "Train Error")
	plt.plot(hist["epoch"], hist["val_mean_absolute_error"],
		   label = "Val Error")
	plt.legend()
	
	plt.figure()
	plt.xlabel("Epoch")
	plt.ylabel("Mea Square Error [$Resistance^2$]")
	plt.plot(hist["epoch"], hist["mean_squared_error"],
			label = "Train Error")
	plt.plot(hist["epoch"], hist["val_mean_squared_error"],
			label = "Val Error")
	plt.legend()
	plt.show()
	
# plot_history(history)

model = build_model()

# Stop and check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
				   validation_split = 0.2, verbose = 0, 
				   callbacks = [early_stop, PrintDot()])

# plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 0)

print("Testing set mean abs error: {:5.4f}".format(mae))
# print("Testing set mean squ error: {:5.2f}".format(mse)+var_testing)

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values"+var_testing)
plt.ylabel("Predictions"+var_testing)
plt.axis("equal")
plt.axis("square")
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-1500, 1500], [-1500, 1500])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error"+var_testing)
_ = plt.ylabel("Count")
# plt.show()
end = time.time() 
print("Mac time: ", end - start)