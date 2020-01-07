import numpy as np 
import matplotlib.pyplot as plt 
import random 

class LMS():

	def __init__(self, dataset, rate):

		self.dataset = dataset 
		self.rate = rate

		self.data = []
		with open(self.dataset, 'r') as f:
			for line in f:
				self.data.append([float(item) for item in line.strip().split(',')])

		self.cost = []
		self.sizing = 1

	def normalize_data(self):

		mini, maxi = np.ones(len(self.data[0]))*1000000, np.zeros(len(self.data[0]))
		for j in range(len(self.data)):
			for jj in range(len(self.data[0])):
				if self.data[j][jj] > maxi[jj]:
					maxi[jj] = self.data[j][jj]
				if self.data[j][jj] < mini[jj]:
					mini[jj] = self.data[j][jj] 

		for j in range(len(self.data)):
			for jj in range(len(self.data[0])):
				self.data[j][jj] = (self.data[j][jj] - mini[jj])/(maxi[jj]-mini[jj])

	def setup_weights(self):

		self.weights = np.zeros((len(self.data[0])-1))

	def setup_biases(self):

		self.bias = 0

	def dot_things(self, weight, x):

		wtx = 0
		for j in range(len(weight)):
			wtx += weight[j]*x[j] 

		return wtx

	def compute_gradient_of_Jw(self):

		self.Jw = 0 
		inter_cost = []
		for j in range(len(self.data)-1):
			mult = sum(self.weights*np.array(self.data[j])[0:11])
			for jj in range(len(self.data[0])-1):
				inter_cost.append((1/2)*(self.data[j][-1] - mult - self.bias)**2)
				self.weights[jj] += self.rate*(self.data[j][-1] - mult)*self.data[j][jj]
				self.bias += self.rate*(self.data[j][-1] - mult - self.bias) 

		self.cost.append(np.mean(inter_cost))

	def batch_gradient_descent(self, tol):

		self.error = 1
		self.normalize_data()
		self.setup_weights()
		self.setup_biases()

		self.num = 0
		while self.error > tol and self.num < 1000:
			self.compute_gradient_of_Jw()
			self.num += 1

		self.which_variables_affect_sensor()
		# self.plot_stuff()

	def which_variables_affect_sensor(self):

		ordered_weights = list(self.weights)
		count, N = 0, 0
		while count < len(self.weights) - 1 and N < 1000:
			for j in range(len(self.weights)-1):
				if abs(ordered_weights[j]) > abs(ordered_weights[j+1]):
				# if ordered_weights[j] > ordered_weights[j+1]:
					ordered_weights[j], ordered_weights[j+1] = ordered_weights[j+1], ordered_weights[j]
				else:
					count += 1
			count = 0
			N += 1

		# col_names = ["Wind Speed", "Wind Direction", "Temp", "SO2", "Relative Humidity", "PM2.5", "O3", "NOX", "NO2", "NO", "CO"]
		col_names = ["Wind Speed", "Wind Direction", "Temp", "SO2", "Relative Humidity", "PM2.5", "NOX", "NO2", "NO", "CO", "AirU"]

		cat = []
		for j in range(len(ordered_weights)):
			for jj in range(len(self.weights)):
				if ordered_weights[j] == self.weights[jj]:
					cat.append(col_names[jj])

		print(cat[::-1])

	def plot_stuff(self):

		plt.plot(self.cost)
		plt.show()
	
	def print_stuff(self):
		print('weights: ', self.weights)
		print('bias: ', self.bias)

tol = 1e-6
rate = 0.001 
# datasets = ["HW137-92218-92818_data.csv", "HW103-92218-92818_data.csv", "HW016-92218-92818_data.csv"]
# datasets = ["HW137-92218-92818_O3data.csv", "HW103-92218-92818_O3data.csv", "HW016-92218-92818_O3data.csv"]
# print('')
# print('Ranked categories for each sensor')
# for j in range(len(datasets)):
# 	LMS(datasets[j], rate).batch_gradient_descent(tol)

# print('')

