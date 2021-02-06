import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from parseData import *

class baseLSTM(nn.Module):
	def __init__(self, units, layers, batch_size, drp_rate, channels, classes):
		super(baseLSTM, self).__init__()
		self.layers = layers
		self.units = units
		self.hs = self.init_hidden(batch_size)

		self.lstm = nn.LSTM(channels, units, layers, batch_first=True, dropout=drp_rate)
		self.linear = nn.Linear(units, classes)

	def forward(self, input_data):
		x, self.hs = self.lstm(input_data, self.hs)
		y = self.linear(x[:, -1, :])
		z = F.log_softmax(y, 1)
		return z

	def init_hidden(self, batch_size):
		hx = nn.init.orthogonal_(torch.Tensor(self.layers, batch_size, self.units))
		cx = nn.init.orthogonal_(torch.Tensor(self.layers, batch_size, self.units))
		return (hx.cuda(), cx.cuda())

	def preserve_state(self, prv_hs):
		hx = prv_hs[0].detach()
		cx = prv_hs[1].detach()
		return (hx, cx)


class enseLSTM(nn.Module):
	def __init__(self, total_lstms, classes):
		super(enseLSTM, self).__init__()
		self.linear1 = nn.Linear(total_lstms * classes, 128)
		self.linear2 = nn.Linear(128, 64)
		self.linear3 = nn.Linear(64, classes)

	def forward(self, input_data):
		x = F.relu(self.linear1(input_data.view(1, -1)))
		y = F.relu(self.linear2(x))
		z = F.log_softmax(self.linear3(y), 1)
		return z


class harDataset(data.Dataset):
	def __init__(self, track, subjects, runs, wl, ws):
		self.win_len = wl
		self.win_stp = ws
		self.x, self.y, self.checkpoints = load_data(track, subjects, runs, wl, ws)
		self.len = len(self.y)

	def __len__(self):
		return self.len

	def __getitem__(self, win_idx):
		x = self.x[win_idx * self.win_len : (win_idx + 1) * self.win_len]
		y = self.y[win_idx]
		return x, y


class harSubset(data.Subset):
	def __init__(self, dataset, indices):
		self.dataset = dataset
		self.indices = indices

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		x, y = self.dataset[self.indices[idx]]
		return x, y
