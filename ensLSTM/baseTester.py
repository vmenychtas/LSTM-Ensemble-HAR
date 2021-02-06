import torch
import torch.nn as nn
import torch.utils.data as data
from nnClasses import *
from parseData import *
from nnMetrics import *
import pandas as pd
import argparse
import time
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-id', '--lstm_id', required=True, help='Model ID')
args = vars(parser.parse_args())

id = int(args['lstm_id'])

cfg = pd.read_csv('models.cfg', delimiter=' ', decimal='.', header=None)
cfg = cfg[cfg[0] == id]
if cfg.shape[0] is 0:
	sys.exit('Error: Model configuration not found.\n')
else:
	ix = cfg.values[0][1]
	UNITS = cfg.values[0][2]
	WIN_LEN = cfg.values[0][3]

BATCH = 1
LAYERS = 2
SENSORS = 77

TRACK_LIST = {1: 'Locomotion',
			  2: 'HL_Activity',
			  3: 'LL_Left_Arm',
			  4: 'LL_Left_Arm_Object',
			  5: 'LL_Right_Arm',
			  6: 'LL_Right_Arm_Object',
			  7: 'ML_Both_Arms'}

index, dictionary, CLASSES_NUM = read_label_legend(TRACK_LIST[ix])

SUBJECTS = [2, 3]

RUNS = {2: [4, 5],
		3: [4, 5]}

#--------------------- Instantiate baseLSTM model ---------------------#
lstm = baseLSTM(UNITS, LAYERS, BATCH, 0, SENSORS, CLASSES_NUM).cuda()

state_file = 'parameters/b' + str(id) + '.pt'
if os.path.isfile(state_file):		# Load model's saved state
	lstm.load_state_dict(torch.load(state_file))
else:
	sys.exit('Error: Could not recover model parameters.\n')

lstm.eval()
print('- Model initialization complete.')

#-------------------------- Load the dataset --------------------------#
testset = harDataset(ix, SUBJECTS, RUNS, WIN_LEN, WIN_LEN)

#------------------------ Prepare the subsets -------------------------#
total_windows = 0
testRuns = {}

start = 0 ; r = 0
for sub in SUBJECTS:
	testRuns[sub] = {}
	for run in RUNS[sub]:
		end = start + testset.checkpoints[r]
		testRuns[sub][run] = harSubset(testset, np.arange(start, end).tolist())
		total_windows += len(testRuns[sub][run])
		start = end ; r += 1

print('- Test set: {} samples'.format(total_windows))

#---------------------------- Run the test ----------------------------#
criterion = nn.NLLLoss().cuda()
total_loss = 0.0

prediction = []
truth = []

print('Running test..\n')
start = time.time()

for sub in SUBJECTS:
	for run in RUNS[sub]:
		lstm.hs = lstm.init_hidden(1)		 # Initialize models hidden state
		loader = data.DataLoader(testRuns[sub][run])

		for count, win in enumerate(loader):
			lstm.hs = lstm.preserve_state(lstm.hs)		# Retain hidden state

			data_in = win[0].detach()
			target = prepare_targets(win[1])

			out = lstm(data_in)
			loss = criterion(out, target)
			total_loss += loss.item()

			_, pred = torch.max(out, 1)
			truth += target.data.tolist()
			prediction += pred.data.tolist()

test_time = round(time.time() - start)		# Inference process complete

average_loss = total_loss / total_windows
accuracy = calc_accuracy(truth, prediction)
f1, f1_weight = f1_score(truth, prediction, CLASSES_NUM)

print(' F1 weighted: {0:5.3f}'.format(f1_weight))
print('          F1: {0:5.3f}'.format(f1))
print('    Accuracy: {0:4.1f} %'.format(accuracy * 100))
print('        Loss: {0:5.3f}'.format(average_loss))
print('\nTesting complete. ({} sec)'.format(test_time))

res_file = 'results.dat'
if not os.path.isfile(res_file):		# Save test results to file
	with open(res_file, 'w') as f:
		f.write('Model_ID F1_weighted F1 Accuracy Loss\n')
		f.write('{0} {1:5.3f} {2:5.3f} {3:4.1f} {4:5.3f}\n'
		 .format(id, f1_weight, f1, accuracy * 100, average_loss))
else:
	with open(res_file, 'a') as f:
		f.write('{0} {1:5.3f} {2:5.3f} {3:4.1f} {4:5.3f}\n'
		 .format(id, f1_weight, f1, accuracy * 100, average_loss))
