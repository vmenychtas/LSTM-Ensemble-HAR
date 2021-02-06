import torch
import numpy as np
import pandas as pd
import random

def read_label_legend(track_name):
	activities = ['NULL']
	unique_index = [0]		# 0 represents the NULL class
	path = "../data/OpdUCI/dataset/label_legend.txt"
	f = open(path).readlines()
	# omit the first two lines
	line1 = f.pop(0) ; line2 = f.pop(0)

	for line in f:
		line = line.strip("\n").split("   -   ")
		if line[1] == track_name:
			line[0] = int(line[0])
			unique_index.append(line[0])
			activities.append(line[2])

	classes_num = len(unique_index)
	adjusted_index = np.arange(0, classes_num)
	dictionary = dict(zip(adjusted_index, activities))

	return adjusted_index, dictionary, classes_num


def load_file(subject, run, track):
	if run == 0:
		path = '../data/processed/S{}-Drill.dat'.format(subject)
	else:
		path = '../data/processed/S{}-ADL{}.dat'.format(subject, run)

	raw_data = pd.read_csv(path, delimiter=' ', decimal='.', header=None)
	columns = raw_data.shape[1] - 7

	x_data = raw_data.values[:, 0:columns]
	y_data = raw_data.values[:, columns + track - 1]

	return x_data, y_data


def sliding_window_partition(x_data, y_data, win_len, step=0):
	if step is 0:
		step = win_len // 2

	x_win = []
	y_win = []
	windows = x_data.shape[0] // step - 1

	for w in range(windows):
		# Assign most common label as window label
		labels = y_data[w * step : w * step + win_len].tolist()
		y_win.append(max(set(labels), key=labels.count))
		for k in range(win_len):
			x_win.append(x_data[w * step + k, :])

	return x_win, y_win


def load_data(track, subjects, runs, win_len, win_stp):
	x_data = []
	y_data = []
	checkpoint = []

	for sub in subjects:
		for run in runs[sub]:
			x, y = load_file(sub, run, track)
			win_x, win_y = sliding_window_partition(x, y, win_len, win_stp)

			checkpoint.append(len(win_y))
			x_data += win_x
			y_data += win_y

	x_data = torch.Tensor(np.array(x_data)).cuda()
	y_data = torch.Tensor(np.array(y_data)).cuda()

	return x_data, y_data, checkpoint


def generate_sequences(start, end, seq_len):
	sequence = []
	total_windows = end - start
	num_start_points = total_windows // seq_len

	seq_start_points = []
	for p in range(num_start_points):
		seq_start_points.append(start + p * seq_len)

	random.shuffle(seq_start_points)
	num_start_points = int(num_start_points * 0.9)
	seq_start_points = seq_start_points[0: num_start_points]
	seq_start_points.sort()

	for x in range(num_start_points):
		s = np.arange(seq_start_points[x], seq_start_points[x] + seq_len)
		sequence += s.tolist()

	return sequence, len(sequence)


def prepare_targets(activity):
	length = activity.view(-1).shape[0]
	targets = torch.zeros(length)

	if length == 1:
		targets[0] = activity.item()
		return targets.long().cuda()

	for x in range(length):
		targets[x] = activity[x]

	return targets.long().cuda()
