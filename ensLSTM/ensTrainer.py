import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from nnClasses import *
from parseData import *
from nnMetrics import *
import pandas as pd
import numpy as np
import argparse
import random
import time
import sys
import os

def train(nsmbl, baseLSTMs, total_bm, lrn_rate, nn_scores, epoch):
	optimizer = optim.SGD(nsmbl.parameters(), lr=lrn_rate)
	criterion = nn.NLLLoss().cuda()

	average_loss = {'train': 0.0, 'valid': 0.0}
	total_loss = {'train': 0.0, 'valid': 0.0}
	accuracy = {'train': 0.0, 'valid': 0.0}

	prediction = {'train': [], 'valid': []}
	truth = {'train': [], 'valid': []}

	#--------------- Train Ensemble Classifier ---------------#
	start = time.time()

	random.shuffle(SUBJECTS['train'])
	for sub in SUBJECTS['train']:
		random.shuffle(RUNS['train'][sub])

		for run in RUNS['train'][sub]:
			loader = data.DataLoader(trainRuns[sub][run])
			for m in range(total_bm):
				baseLSTMs[m].hs = baseLSTMs[m].init_hidden(1)

			for count, win in enumerate(loader):
				nsmbl.zero_grad()			# Zero ensemble gradients

				base_out = []
				for m in range(total_bm):	# Retain hidden state for all baseLSTM models
					baseLSTMs[m].hs = baseLSTMs[m].preserve_state(baseLSTMs[m].hs)
					out = baseLSTMs[m](win[0])		# Forward pass
					base_out.append(out)

				x = torch.exp(base_out[0])			# Concatenate base outputs
				for m in range(1, total_bm):
					x = torch.cat((x, torch.exp(base_out[m])), 0)

				x = x.view(1, 1, total_bm, -1).detach()
				target = prepare_targets(win[1])

				ens_out = nsmbl(x)							# Forward pass
				ens_loss = criterion(ens_out, target)	  # Calculate loss
				total_loss['train'] += ens_loss.item()

				ens_loss.backward()					 # Calculate gradients
				optimizer.step()				 # Adjust model parameters

				_, pred = torch.max(ens_out, 1)
				truth['train'] += target.data.tolist()
				prediction['train'] += pred.data.tolist()

	train_time = round(time.time() - start)			# Train epoch complete

	average_loss['train'] = total_loss['train'] / total_windows['train']
	accuracy['train'] = calc_accuracy(truth['train'], prediction['train'])

	#----------------- Process validation set ----------------#
	start = time.time()

	for sub in SUBJECTS['valid']:
		for run in RUNS['valid'][sub]:
			loader = data.DataLoader(validRuns[sub][run])
			for m in range(total_bm):
				baseLSTMs[m].hs = baseLSTMs[m].init_hidden(1)

			for count, win in enumerate(loader):
				base_out = []
				for m in range(total_bm):
					baseLSTMs[m].hs = baseLSTMs[m].preserve_state(baseLSTMs[m].hs)
					out = baseLSTMs[m](win[0])
					base_out.append(out)

				x = torch.exp(base_out[0])
				for m in range(1, total_bm):
					x = torch.cat((x, torch.exp(base_out[m])), 0)

				x = x.view(1, 1, total_bm, -1).detach()
				target = prepare_targets(win[1])

				ens_out = nsmbl(x)
				ens_loss = criterion(ens_out, target)
				total_loss['valid'] += ens_loss.item()

				_, pred = torch.max(ens_out, 1)
				truth['valid'] += target.data.tolist()
				prediction['valid'] += pred.data.tolist()

	total_time = round(train_time + time.time() - start)	# Validation process complete

	average_loss['valid'] = total_loss['valid'] / total_windows['valid']
	accuracy['valid'] = calc_accuracy(truth['valid'], prediction['valid'])
	f1, f1_weight = f1_score(truth['valid'], prediction['valid'], CLASSES_NUM)

	if (f1_weight > nn_scores['f1_w']) or (f1_weight == nn_scores['f1_w'] and f1 > nn_scores['f1_m']):
		nn_scores['f1_w'] = f1_weight
		nn_scores['f1_m'] = f1
		nn_scores['accu'] = accuracy['valid']
		nn_scores['loss'] = average_loss['valid']
		nn_scores['epoch'] = epoch

		torch.save(nsmbl.state_dict(), state_file)	# Save model state to file

		with open(scores_path, 'w') as f:			# Save model's best scores
			f.write('F1_weighted F1_score Valid_accu Valid_loss Epoch\n')
			f.write('{0:5.3f} {1:5.3f} {2:4.1f} {3:5.3f} {4}\n'
			.format(f1_weight, f1, accuracy['valid'] * 100, average_loss['valid'], epoch + 1))

	with open(progress_file, 'a') as f:				# Update model progress file
		f.write('{0:5.3f} {1:5.3f} {2:4.1f} {3:4.1f} {4:5.3f} {5:5.3f}\n'
				.format(average_loss['train'], average_loss['valid'], accuracy['train'] * 100,
				  		accuracy['valid'] * 100, f1, f1_weight))

	print('[Epoch {0:2d}]  {1:5.3f} | {2:5.3f}{8}{3:4.1f} % | {4:4.1f} %{8}{5:5.3f} | {6:5.3f}  ({7}{9}'
		  .format(epoch+1,average_loss['train'],average_loss['valid'],accuracy['train']*100,				  accuracy['valid']*100,f1,f1_weight,train_time,'  --  ',' sec)'))

	return nn_scores


#----------------------------------------------------------------------#
#----------------------- Parse script arguments -----------------------#
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--nsmbl_id', required=True,  help='Model ID')
parser.add_argument('-ca', '--cat_act',	 required=False, help='Category of activities')
parser.add_argument('-li', '--id_list',  required=False, help='List of baseLSTM models')
parser.add_argument('-lr', '--lr_rate',	 required=False, help='Learning rate')
parser.add_argument('-ep', '--epochs',	 required=False, help='Number of epochs')
args = vars(parser.parse_args())

id = int(args['nsmbl_id'])

if args['id_list'] is not None:		# Use list given
	list_provided = True
	tokens = args['id_list'].split(',')
	id_list = [int(s) for s in tokens]
elif args['cat_act'] is not None:	# Use category given
	list_provided = False
	id_list = []
	ix = int(args['cat_act'])
	if ix < 1 or ix > 7:
		sys.exit('Invalid category [index out of bounds]')
else:
	sys.exit('Error: No model list or category of activities specified.\n')

LRN_RATE = 0.01
if args['lr_rate'] is not None:
	LRN_RATE = float(args['lr_rate'])

epochs = 10
if args['epochs'] is not None:
	epochs = int(args['epochs'])

LAYERS = 2
SENSORS = 77		# Number of sensor channels

TRACK_LIST = {1: 'Locomotion',
			  2: 'HL_Activity',
			  3: 'LL_Left_Arm',
			  4: 'LL_Left_Arm_Object',
			  5: 'LL_Right_Arm',
			  6: 'LL_Right_Arm_Object',
			  7: 'ML_Both_Arms'}

SUBJECTS = {'train': [1, 2, 3, 4],
			'valid': [1, 2, 3, 4]}

RUNS = {'train': {1: [0, 2, 4, 5], 2: [0, 1, 2, 3],
				  3: [0, 1, 2, 3], 4: [0, 1, 3, 5]},
		'valid': {1: [1, 3], 2: [], 3: [], 4: [2, 4]}}

# RUNS = {'train': {1: [2, 4], 2: [1, 3], 3: [2, 3], 4: [1, 5]},
# 		'valid': {1: [1, 3], 2: [],		3: [],	   4: [2]}}

#------------------- Retrieve ensemble configuration ------------------#
config_file = 'nsmbls.cfg'
prv_config = False

if os.path.isfile(config_file):		# Check if ensemble has already been configured
	cfg = pd.read_csv(config_file, delimiter=' ', decimal='.', header=None)
	cfg = cfg[cfg[0] == id].reset_index(drop=True)

	if not cfg.shape[0] == 0:		# Ensemble previously configured; recover data
		ix = cfg[1][0]
		total_models = cfg[2][0]
		CLASSES_NUM = cfg[3][0]
		WIN_LEN = cfg[4][0]
		LRN_RATE = cfg[5][0]
		if args['epochs'] is None:
			epochs = cfg[6][0]
		id_list = cfg[7][0].split(',')
		id_list = [int(s) for s in id_list]

		print('Ensemble has been previously configured.\n')
		prv_config = True

#---------------- Retrieve base models' configurations ----------------#
if not os.path.isfile('models.cfg'):
	sys.exit('Error: Base models configuration file not found.\n')
else:
	cfg_base = pd.read_csv('models.cfg', delimiter=' ', decimal='.', header=None)

#-------------------- Instantiate baseLSTM models ---------------------#
baseModels = nn.ModuleList()
cfg = {'id_list': [], 'UNITS': [], 'categories': []}

if list_provided and not prv_config:
	for m in range(len(id_list)):
		temp = cfg_base[cfg_base[0] == int(id_list[m])].reset_index(drop=True)
		cfg['UNITS'].append(temp.values[0][2])
		cfg['categories'].append(temp.values[0][1])
	cfg['id_list'] = id_list

	# Verify models aren't trained on different class categories
	if len(set(cfg['categories'])) is 1:
		ix = cfg['categories'][0]
		index, dictionary, CLASSES_NUM = read_label_legend(TRACK_LIST[ix])

		for m in range(len(id_list)):
			baseModels.append(baseLSTM(cfg['UNITS'][m], LAYERS, 1, 0, SENSORS, CLASSES_NUM).cuda())
	else:
		sys.exit('Error: Specified models have been trained on different categories.\n')

elif not prv_config:		# No IDs specified; use all models trained on specified category
	index, dictionary, CLASSES_NUM = read_label_legend(TRACK_LIST[ix])

	for m in range(len(cfg_base)):
		if cfg_base.values[m][1] == ix:
			cfg['categories'].append(cfg_base.values[m][1])
			cfg['id_list'].append(cfg_base.values[m][0])
			cfg['UNITS'].append(cfg_base.values[m][2])
			baseModels.append(baseLSTM(cfg['UNITS'][m], LAYERS, 1, 0, SENSORS, CLASSES_NUM).cuda())

else:						# Ensemble previously configured
	_, dictionary, _ = read_label_legend(TRACK_LIST[ix])

	cfg['id_list'] = id_list
	for m in range(len(id_list)):
		temp = cfg_base[cfg_base[0] == int(id_list[m])].reset_index(drop=True)
		cfg['UNITS'].append(temp.values[0][2])
		cfg['categories'].append(temp.values[0][1])
		baseModels.append(baseLSTM(cfg['UNITS'][m], LAYERS, 1, 0, SENSORS, CLASSES_NUM).cuda())

if cfg['categories'][0] is 1:
	WIN_LEN = 40
elif cfg['categories'][0] is 7:
	WIN_LEN = 30

total_models = len(baseModels)

#-------------------- Store ensemble configuration --------------------#
if not prv_config:
	model_list = str(cfg['id_list'][0])
	for x in range(1, total_models):
		model_list = model_list + ',' + str(cfg['id_list'][x])

	with open(config_file, 'a') as f:
		f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'
		.format(id, ix, total_models, CLASSES_NUM, WIN_LEN, LRN_RATE, epochs, model_list))

print('- Activity track chosen: {}'.format(TRACK_LIST[ix]))
print('- Total number of classes: {}'.format(CLASSES_NUM))
print('- Total number of baseLSTMs: {}'.format(total_models))
print('  Models selected: {}'.format(cfg['id_list']))

#------------------ Load saved states of base models ------------------#
for m in range(total_models):
	baseModels[m].load_state_dict(torch.load('parameters/b'+ str(cfg['id_list'][m]) +'.pt'))
	baseModels[m].eval()

#-------------------------- Load the dataset --------------------------#
trainset = harDataset(ix, SUBJECTS['train'], RUNS['train'], WIN_LEN, WIN_LEN)
validset = harDataset(ix, SUBJECTS['valid'], RUNS['valid'], WIN_LEN, WIN_LEN)

#------------------------ Prepare the subsets -------------------------#
total_windows = {'train': 0, 'valid': 0}
trainRuns = {}
validRuns = {}

start = 0 ; r = 0
for sub in SUBJECTS['train']:
	trainRuns[sub] = {}
	for run in RUNS['train'][sub]:
		end = start + trainset.checkpoints[r]
		trainRuns[sub][run] = harSubset(trainset, np.arange(start, end).tolist())
		total_windows['train'] += len(trainRuns[sub][run])
		start = end ; r += 1

start = 0 ; r = 0
for sub in SUBJECTS['valid']:
	validRuns[sub] = {}
	for run in RUNS['valid'][sub]:
		end = start + validset.checkpoints[r]
		validRuns[sub][run] = harSubset(validset, np.arange(start, end).tolist())
		total_windows['valid'] += len(validRuns[sub][run])
		start = end ; r += 1

print('- Training set: {} samples | Validation set: {} samples'
	  .format(total_windows['train'], total_windows['valid']))

#----------------------- Instantiate Ensemble  ------------------------#
ensemble = enseLSTM(total_models, CLASSES_NUM).cuda()

progress_file = 'scores/n' + str(id) + '_prog'
scores_path = 'scores/n' + str(id) + '_best'
scores = {'f1_w': 0.0,
 		  'f1_m': 0.0,
		  'accu': 0.0,
		  'loss': 0.0,
		  'epoch': 0}

state_file = 'parameters/n' + str(id) + '.pt' 	# Load ensemble saved
if os.path.isfile(state_file):					# state, if it exists
	ensemble.load_state_dict(torch.load(state_file))

offset = 0									  # Epochs previously trained
if os.path.isfile(progress_file):
	offset = pd.read_csv(progress_file).shape[0]

if offset == 0:
	with open(progress_file, 'w') as f:
		f.write('Train_loss Valid_loss Train_accu Valid_accu F1_score F1_weighted\n')

if os.path.isfile(scores_path):	  # Recover previous scores, if they exist
	saved_scores = pd.read_csv(scores_path, delimiter=' ', decimal='.')
	scores['f1_w'] = saved_scores.values[0][0]
	scores['f1_m'] = saved_scores.values[0][1]
	scores['accu'] = saved_scores.values[0][2]
	scores['loss'] = saved_scores.values[0][3]
	scores['epoch'] = saved_scores.values[0][4].astype(int)

#--------------------- Initiate training process ----------------------#
while True:
	mode = input('\nWhat would you like to do? [ (\033[4mT\033[0m)rain / (Q)uit ] : ')

	if mode == 't' or mode == 'T' or mode == '':	# Train the model
		print('\n{0:>20s}{1:>23s}{2:>20s}'.format('Loss', 'Accuracy', 'F1 Score'))

		for e in range(offset, epochs + offset):
			scores = train(ensemble, baseModels, total_models, LRN_RATE, scores, e)

		offset = e + 1
	elif mode == 'q' or mode == 'Q':	#------------ Quit the script
		print('Done!')
		break
	else:
		print('Invalid input; try again.')
