import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from nnClasses import *
from parseData import *
from nnMetrics import *
import numpy as np
import argparse
import select
import random
import pickle
import time
import sys
import os

def train(model, batch, lrn_rate, nn_scores, epoch):
	optimizer = optim.SGD(model.parameters(), lr=lrn_rate)
	criterion = nn.NLLLoss().cuda()

	average_loss = {'train': 0.0, 'valid': 0.0}
	total_loss = {'train': 0.0, 'valid': 0.0}
	accuracy = {'train': 0.0, 'valid': 0.0}

	prediction = {'train': [], 'valid': []}
	truth = {'train': [], 'valid': []}

	#--------------- Train baseLSTM classifier --------------#
	model.train()
	start = time.time()

	random.shuffle(SUBJECTS['train'])
	for sub in SUBJECTS['train']:
		random.shuffle(RUNS['train'][sub])

		for run in RUNS['train'][sub]:
			model.hs = model.init_hidden(batch)	 # Initialize models hidden state
			loader = data.DataLoader(trainRuns[sub][run], batch_size=batch, drop_last=True)

			for count, win in enumerate(loader):
				model.zero_grad()						   # Zero model gradients
				model.hs = model.preserve_state(model.hs)	# Retain hidden state

				if flpcoin and (count % SEQ_LEN) is 0:
					if random.choice((0, 1)) is 0:
						model.hs = model.init_hidden(batch)

				data_in = win[0].detach()
				target = prepare_targets(win[1])

				out = model(data_in)				   # Forward pass
				loss = criterion(out, target)		 # Calculate loss
				total_loss['train'] += loss.item()

				loss.backward()					# Calculate gradients
				optimizer.step()			# Adjust model parameters

				_, pred = torch.max(out, 1)
				truth['train'] += target.data.tolist()
				prediction['train'] += pred.data.tolist()

	train_time = round(time.time() - start)	   # Train epoch complete

	average_loss['train'] = total_loss['train'] / total_windows['train']
	accuracy['train'] = calc_accuracy(truth['train'], prediction['train'])

	#---------------- Process validation set ----------------#
	model.eval()
	start = time.time()

	for sub in SUBJECTS['valid']:
		for run in RUNS['valid'][sub]:
			model.hs = model.init_hidden(1)		 # Initialize models hidden state
			loader = data.DataLoader(validRuns[sub][run])

			for count, win in enumerate(loader):
				model.hs = model.preserve_state(model.hs)	# Retain hidden state

				data_in = win[0].detach()
				target = prepare_targets(win[1])

				out = model(data_in)
				loss = criterion(out, target)
				total_loss['valid'] += loss.item()

				_, pred = torch.max(out, 1)
				truth['valid'] += target.data.tolist()
				prediction['valid'] += pred.data.tolist()

	total_time = round(train_time + time.time() - start)	# Validation process complete

	average_loss['valid'] = total_loss['valid'] / total_windows['valid']
	accuracy['valid'] = calc_accuracy(truth['valid'], prediction['valid'])
	f1, f1_weight = f1_score(truth['valid'], prediction['valid'], CLASSES_NUM)

	if f1_weight > nn_scores['f1_w'] or f1_weight == nn_scores['f1_w'] and f1 > nn_scores['f1_m']:
		nn_scores['f1_w'] = f1_weight
		nn_scores['f1_m'] = f1
		nn_scores['accu'] = accuracy['valid']
		nn_scores['loss'] = average_loss['valid']
		nn_scores['epoch'] = epoch

		torch.save(model.state_dict(), state_file)	# Save model state to file

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
parser.add_argument('-id', '--lstm_id',	required=True,  help='Model ID')
parser.add_argument('-ca', '--cat_act',	required=True,  help='Category of activities')
parser.add_argument('-un', '--units',	required=False,	help='Units per layer')
parser.add_argument('-ba', '--batch',	required=False,	help='Batch size')
parser.add_argument('-wl', '--winlen',	required=False, help='Window length')
parser.add_argument('-ws', '--winstp',	required=False, help='Sliding window step')
parser.add_argument('-lr', '--lr_rate', required=False,	help='Learning rate')
parser.add_argument('-dr', '--dropout',	required=False, help='Dropout rate')
parser.add_argument('-sq', '--seqlen',	required=False, help='Windows sequence length')
parser.add_argument('-fc', '--flpcoin',	required=False, help='Flip coin to reset hidden state')
parser.add_argument('-ep', '--epochs',	required=False, help='Number of epochs')
args = vars(parser.parse_args())

id = int(args['lstm_id'])

ix = int(args['cat_act'])
if ix < 1 or ix > 7:
	sys.exit('Invalid category [index out of bounds]')

if args['units'] is not None:
	UNITS = int(args['units'])
else:
	UNITS = 100

if args['batch'] is not None:
	BATCH = int(args['batch'])
else:
	BATCH = 10

if args['winlen'] is not None:
	WIN_LEN = int(args['winlen'])
elif ix is 1:
	WIN_LEN = 40
else:
	WIN_LEN = 30	# 30 samples = 1 sec

if args['winstp'] is not None:
	WIN_STP = int(args['winstp'])
else:				# 50% window overlap
	WIN_STP = WIN_LEN // 2

if args['lr_rate'] is not None:
	LRN_RATE = float(args['lr_rate'])
else:
	LRN_RATE = 0.01

if args['dropout'] is not None:
	DRP_RATE = float(args['dropout'])
else:
	DRP_RATE = 0.5

if args['seqlen'] is not None:
	SEQ_LEN = int(args['seqlen'])
else:
	SEQ_LEN = 60	# Consecutive windows before coin flip

if args['flpcoin'] is not None:
	flpcoin = args['flpcoin']
else:
	flpcoin = True

if args['epochs'] is not None:
	epochs = int(args['epochs'])
else:
	epochs = 50

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

index, dictionary, CLASSES_NUM = read_label_legend(TRACK_LIST[ix])

print('- Activity track chosen: {}'.format(TRACK_LIST[ix]))
print('- Total number of classes: {}'.format(CLASSES_NUM))

if not os.path.isdir('parameters/'):
	os.mkdir('parameters/')
if not os.path.isdir('scores/'):
	os.mkdir('scores/')
if not os.path.isdir('sequences/'):
	os.mkdir('sequences/')

#---------------- Store / retrieve model configuration ----------------#
config_file = 'models.cfg'
if not os.path.isfile(config_file):			# Create models' config file
	with open(config_file, 'w') as f:		# Save model hyperparameters
		f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'
		.format(id,ix,UNITS,BATCH,WIN_LEN,WIN_STP,LRN_RATE,DRP_RATE,SEQ_LEN,flpcoin,epochs))
else:										# Config file already exists
	cfg = pd.read_csv(config_file, delimiter=' ', decimal='.', header=None)
	cfg = cfg[cfg[0] == id]					# Search baseLSTM ID in file
	if not cfg.shape[0] is 0:				# Model exists; recover hyperparameters
		ix = cfg.values[0][1]
		UNITS = cfg.values[0][2]
		BATCH = cfg.values[0][3]
		WIN_LEN = cfg.values[0][4]
		WIN_STP = cfg.values[0][5]
		if args['lr_rate'] is None:
			LRN_RATE = cfg.values[0][6]
		DRP_RATE = cfg.values[0][7]
		SEQ_LEN = cfg.values[0][8]
		flpcoin = cfg.values[0][9]
		if args['epochs'] is None:
			epochs = cfg.values[0][10]
	else:									# Model not found; save hyperparameters
		with open(config_file, 'a') as f:
			f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'
			.format(id,ix,UNITS,BATCH,WIN_LEN,WIN_STP,LRN_RATE,DRP_RATE,SEQ_LEN,flpcoin,epochs))

#-------------------------- Load the dataset --------------------------#
trainset = harDataset(ix, SUBJECTS['train'], RUNS['train'], WIN_LEN, WIN_STP)
validset = harDataset(ix, SUBJECTS['valid'], RUNS['valid'], WIN_LEN, WIN_LEN)

#----- Generate / retrieve window sequences specific to the model -----#
if os.path.isfile('sequences/b' + str(id) + '.pckl'):		# Retrieve sequences
	with open('sequences/b' + str(id) + '.pckl', 'rb') as handle:
		sequences = pickle.load(handle)
else:									# File doesn't exist; generate sequences
	sequences = {}
	start = 0 ; r = 0
	for sub in SUBJECTS['train']:
		sequences[sub] = {}
		for run in RUNS['train'][sub]:
			end = sum(trainset.checkpoints[: r + 1])
			sequences[sub][run], _ = generate_sequences(start, end, SEQ_LEN)
			start = end ; r += 1
	# Save sequences and total windows to file
	with open('sequences/b' + str(id) + '.pckl', 'wb') as handle:
		pickle.dump(sequences, handle, protocol=pickle.HIGHEST_PROTOCOL)

#------------------------ Prepare the subsets -------------------------#
total_windows = {'train': 0, 'valid': 0}
trainRuns = {}
validRuns = {}

for sub in SUBJECTS['train']:
	trainRuns[sub] = {}
	for run in RUNS['train'][sub]:
		trainRuns[sub][run] = harSubset(trainset, sequences[sub][run])
		total_windows['train'] += len(trainRuns[sub][run]) // 10

start = 0 ; r = 0
for sub in SUBJECTS['valid']:
	validRuns[sub] = {}
	for run in RUNS['valid'][sub]:
		end = start + validset.checkpoints[r]
		validRuns[sub][run] = harSubset(validset, np.arange(start, end).tolist())
		total_windows['valid'] += len(validRuns[sub][run])
		start = end ; r += 1

print('- Training set: {} samples | Validation set: {} samples'
	  .format(total_windows['train'] * 10, total_windows['valid']))

#--------------------- Instantiate baseLSTM model ---------------------#
lstm = baseLSTM(UNITS, LAYERS, BATCH, DRP_RATE, SENSORS, CLASSES_NUM).cuda()

progress_file = 'scores/b' + str(id) + '-prog'
scores_path = 'scores/b' + str(id) + '-best'
scores = {'f1_w': 0.0,
 		  'f1_m': 0.0,
		  'accu': 0.0,
		  'loss': 0.0,
		  'epoch': 0}

state_file = 'parameters/b' + str(id) + '.pt' 	# Load model's saved
if os.path.isfile(state_file):					# state if it exists
	lstm.load_state_dict(torch.load(state_file))

offset = 0								 # Epochs previously trained
if os.path.isfile(progress_file):
	offset = pd.read_csv(progress_file).shape[0]

if offset == 0:
	with open(progress_file, 'w') as f:
		f.write('Train_loss Valid_loss Train_accu Valid_accu F1_score F1_weighted\n')

if os.path.isfile(scores_path):			 # Recover previous scores, if they exist
	saved_scores = pd.read_csv(scores_path, delimiter=' ', decimal='.')
	scores['f1_w'] = saved_scores.values[0][0]
	scores['f1_m'] = saved_scores.values[0][1]
	scores['accu'] = saved_scores.values[0][2]
	scores['loss'] = saved_scores.values[0][3]
	scores['epoch'] = saved_scores.values[0][4].astype(int)

#--------------------- Initiate training process ----------------------#
while True:
	print('\nWhat would you like to do? [ (\033[4mT\033[0m)rain / (Q)uit ]')
	s, _, _ = select.select([sys.stdin], [], [], 10)	# Wait 10 sec for input
	if (s):
		mode = sys.stdin.readline().strip()
	elif offset is 0:
		mode = 't'
	else:
		mode = 'q'

	if mode == 't' or mode == 'T' or mode == '':	# Train the model
		print('\n{0:>20s}{1:>23s}{2:>20s}'.format('Loss', 'Accuracy', 'F1 Score'))

		for e in range(offset, epochs + offset):
			scores = train(lstm, BATCH, LRN_RATE, scores, e)

		offset = e + 1
	elif mode == 'q' or mode == 'Q':	#------------ Quit the script
		print('Done!')
		break
	else:
		print('Invalid input; try again.')
