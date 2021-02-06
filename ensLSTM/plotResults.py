import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plt_test_pred(id):
	data = pd.read_csv("results/prediction.dat", delimiter=" ", header=None)
	length = data.values.shape[1]
	prediction = data.values[0, 0:length]

	data = pd.read_csv("results/truth.dat", delimiter=" ", header=None)
	truth = data.values[0, 0:length]

	x_axis = np.arange(length)

	f, ax = plt.subplots(2, sharex=True)
	ax[0].plot(x_axis, truth)
	ax[0].set_title('Truth')
	ax[1].plot(x_axis, prediction)
	ax[1].set_title('Prediction')

	plt.show()


def plt_train_prog(id):
	df = pd.read_csv("base_713_prog", delimiter=" ", decimal=".")
	epochs = np.arange(df.values.shape[0])

	f, ax = plt.subplots(3, sharex=True)

	ax[0].plot(epochs, 'Train_loss', data=df, color='blue')
	ax[0].plot(epochs, 'Valid_loss', data=df, color='red')
	ax[0].legend()

	ax[1].plot(epochs, 'Train_accu', data=df, color='blue')
	ax[1].plot(epochs, 'Valid_accu', data=df, color='red')
	ax[1].legend()

	ax[2].plot(epochs, 'F1_score', data=df, color='blue')
	ax[2].plot(epochs, 'F1_weighted', data=df, color='red')
	ax[2].legend()

	plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('-id', '--modelID',	required=True,  help="Model ID")
args = vars(parser.parse_args())
mID = int(args['modelID'])

opt1 = "\n [\033[4m1\033[0m] Plot training metrics proression"
opt2 = "\n [2] Plot test prediction\n"
msg = "What would you like to do?" + opt1 + opt2

x = input(msg)

if x is 1:
	plt_train_prog(mID)
elif x is 2:
	plt_test_pred(mID)
else:
	print("Wrong input. Exiting")
