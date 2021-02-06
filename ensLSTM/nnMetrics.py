import numpy as np
import warnings

warnings.simplefilter("error", RuntimeWarning)

def calc_accuracy(truth, prediction):
	length = len(truth)
	if (length != len(prediction)):
		print("Lengths not matching.")
		return 0
	return sum([truth[i] == prediction[i] for i in range(length)]) / length


def f1_score(truth, prediction, classes):
	N = len(truth)
	Nc = np.zeros(classes, dtype=int)

	false_pos = np.zeros(classes, dtype=int)
	false_neg = np.zeros(classes, dtype=int)
	true_pos = np.zeros(classes, dtype=int)
	true_neg = np.zeros(classes, dtype=int)

	precision = np.zeros(classes)
	recall = np.zeros(classes)

	for i in range(N):
		t = truth[i]
		p = prediction[i]
		Nc[t] += 1

		if t == p:
			true_pos[t] += 1
			for j in range(classes):
				if j != t:
					true_neg[j] += 1
		elif t != p:
			false_pos[p] += 1
			false_neg[t] += 1
			for j in range(classes):
				if j != t and j != p:
					true_neg[j] += 1

	for x in range(classes):
		try:
			precision[x] = true_pos[x] / (true_pos[x] + false_pos[x])
			recall[x] = true_pos[x] / (true_pos[x] + false_neg[x])
		except RuntimeWarning:
			pass

	weighted_f1 = 0.0
	mean_f1 = 0.0
	for x in range(classes):
		try:
			weighted_f1 += (Nc[x]/N) * (precision[x]*recall[x]) / (precision[x]+recall[x])
			mean_f1 += (precision[x]*recall[x]) / (precision[x]+recall[x])
		except RuntimeWarning:
			pass

	weighted_f1 *= 2
	mean_f1 *= 2 / classes
	return mean_f1, weighted_f1
