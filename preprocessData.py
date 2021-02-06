import os
import numpy as np
import pandas as pd
from sklearn import preprocessing as pr

TRACK_LIST = {0: "Locomotion",
			  1: "HL_Activity",
			  2: "LL_Left_Arm",
			  3: "LL_Left_Arm_Object",
			  4: "LL_Right_Arm",
			  5: "LL_Right_Arm_Object",
			  6: "ML_Both_Arms"}

def select_columns(data):			# on-body sensors only
	deselect = np.arange(1, 37)
	deselect = np.concatenate([deselect, np.arange(46, 50)])
	deselect = np.concatenate([deselect, np.arange(59, 63)])
	deselect = np.concatenate([deselect, np.arange(72, 76)])
	deselect = np.concatenate([deselect, np.arange(85, 89)])
	deselect = np.concatenate([deselect, np.arange(98, 102)])
	deselect = np.concatenate([deselect, np.arange(134, 243)])

	return np.delete(data, deselect, 1)


def get_data_index(track_name):
	unique_index = [0]	# 0 represents the NULL class of activity
	path = "data/OpdUCI/dataset/label_legend.txt"
	f = open(path).readlines()
	line1 = f.pop(0)	# omit the first two lines
	line2 = f.pop(0)

	for line in f:
		line = line.strip("\n").split("   -   ")
		if line[1] == track_name:
			line[0] = int(line[0])
			unique_index.append(line[0])

	return unique_index


def process_data(data):
	data = select_columns(data)

	label_count = len(TRACK_LIST)
	sensor_count = data.shape[1] - label_count
	data_x = data[:, 1:sensor_count]							# sensor readings
	data_y = data[:, sensor_count:(sensor_count + label_count)]	# activity labels
	sample_count = data_x.shape[0]

	# Adjust track label index
	for ix in range(label_count):
		track_index = get_data_index(TRACK_LIST[ix])
		index_length = len(track_index)
		adjusted_ix = np.arange(0, index_length)

		for x in range(sample_count):
			for k in range(index_length):
				if data_y[x, ix] == track_index[k]:
					data_y[x, ix] = adjusted_ix[k]

	# Drop every row that contains NaN in any column
	data_x = data_x[~np.isnan(data_x).any(axis=1)]
	sample_count = data_x.shape[0]
	sensor_count -= 1
	# Values standardization (mean = 0, variance = 1)
	for x in range(sensor_count):
		data_x[:, x] = pr.scale(data_x[:, x])

	return data_x, data_y.astype(int), sample_count, sensor_count, label_count


def subject_run_loader(subject, run):
	if run == 0:
		pathname = "data/OpdUCI/dataset/S{}-Drill.dat".format(subject)
		filename = "S{}-Drill.dat".format(subject)
	else:
		pathname = "data/OpdUCI/dataset/S{}-ADL{}.dat".format(subject, run)
		filename = "S{}-ADL{}.dat".format(subject, run)

	print("Processing {}".format(filename))
	data = pd.read_csv(pathname, delimiter=' ', decimal='.', header=None)
	data_x, data_y, sample_count, sensor_count, label_count = process_data(data.values)

	return data_x, data_y, sample_count, sensor_count, label_count, filename


data_dir = "data/processed/"
if not os.path.isdir(data_dir):
	os.mkdir(data_dir)

for subject in range(1, 5):
	for run in range(6):
		x, y, samples, sensors, labels, target_filename = subject_run_loader(subject, run)
		with open(data_dir + target_filename, "w") as f:
			for k in range(samples):
				for i in range(sensors):
					f.write(str(float("{:01.8f}".format(x[k, i]))) + ' ')
				for j in range(labels - 1):
					f.write(str(y[k, j].astype(int)) + ' ')
				f.write(str(y[k, j + 1].astype(int)) + '\n')
