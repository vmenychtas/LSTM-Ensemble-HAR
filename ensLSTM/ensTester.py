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

# parser = argparse.ArgumentParser()
# parser.add_argument('-id', '--nsmbl_id', required=True, help='Model ID')
# args = vars(parser.parse_args())
#
# id = int(args['nsmbl_id'])
id = 9999
cfg = pd.read_csv('nsmbls.cfg', delimiter=' ', decimal='.', header=None)
cfg = cfg[cfg[0] == id]
if cfg.shape[0] is 0:
	sys.exit('Error: Ensemble configuration not found.\n')
else:
	ix = cfg.values[0][1]
	total_models = cfg.values[0][2]
	CLASSES_NUM = cfg.values[0][3]
	WIN_LEN = cfg.values[0][4]
	id_list = cfg.values[0][7].split(',')
	id_list = [int(s) for s in id_list]

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

#---------------- Retrieve base models' configurations ----------------#
if not os.path.isfile('models.cfg'):
	sys.exit('Error: Base models configuration file not found.\n')
else:
	cfg_base = pd.read_csv('models.cfg', delimiter=' ', decimal='.', header=None)

#-------------------- Instantiate baseLSTM models ---------------------#
baseModels = nn.ModuleList()
for m in range(len(id_list)):
	temp = cfg_base[cfg_base[0] == int(id_list[m])].reset_index(drop=True)
	UNITS = temp.values[0][2]
	baseModels.append(baseLSTM(UNITS, LAYERS, 1, 0, SENSORS, CLASSES_NUM).cuda())

for m in range(total_models):
	baseModels[m].load_state_dict(torch.load('parameters/b'+ str(id_list[m]) +'.pt'))
	baseModels[m].eval()

#------------------------ Instantiate Ensemble  -----------------------#
ensemble = enseLSTM(total_models, CLASSES_NUM).cuda()

state_file = 'parameters/n' + str(id) + '.pt'
if os.path.isfile(state_file):		 # Load ensemble's saved state
	ensemble.load_state_dict(torch.load(state_file))
ensemble.eval()

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
criterion = nn.NLLLoss()
total_loss = 0.0

prediction = []
truth = []

print('Running test..')
start = time.time()

for sub in SUBJECTS:
	for run in RUNS[sub]:
		loader = data.DataLoader(testRuns[sub][run])
		for m in range(total_models):
			baseModels[m].hs = baseModels[m].init_hidden(1)

		for count, win in enumerate(loader):
			base_out = []
			for m in range(total_models):
				baseModels[m].hs = baseModels[m].preserve_state(baseModels[m].hs)
				out = baseModels[m](win[0])
				base_out.append(out)

			x = torch.exp(base_out[0])
			for m in range(1, total_models):
				x = torch.cat((x, torch.exp(base_out[m])), 0)

			x = x.view(1, 1, total_models, -1).detach()
			target = prepare_targets(win[1])

			ens_out = ensemble(x)
			ens_loss = criterion(ens_out, target)
			total_loss += ens_loss.item()

			_, pred = torch.max(ens_out, 1)
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

# TODO: write down results to file

# with open('prediction.dat', 'w') as f:
# 	for i in range(len(prediction)):
# 		f.write(str(prediction[i]) + ' ')
# with open('truth.dat', 'w') as f:
# 	for i in range(len(truth)):
# 		f.write(str(int(truth[i])) + ' ')
