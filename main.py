#!/usr/bin/python

import os, sys, getopt
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.nn import init
import math, random
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from PIL import Image
from torch.utils.data import random_split


class SoundDataSet :
	def __init__(self, metadata_file):
		self.df = pd.read_csv(metadata_file, sep=";", names=["path", "start_time", "duration", "frequency_peak"])
		self.df.head()

		ds_path = os.path.dirname(os.path.abspath(metadata_file))
		self.df['classID'] = np.array([c.split('--')[0] for c in self.df['path']])
		self.classes = np.unique(self.df['classID'])
		self.df['classID'] = np.array([np.where(self.classes==c)[0] for c in self.df['classID']])
		self.df['path'] = ds_path + '/' + self.df['path'];
		self.df = self.df.sample(int(len(self.df)*0.01), ignore_index=True)
		print(len(self.df))
		self.duration = 4000
		self.sr = 44100
		self.channel = 2
		self.shift_pct = 0.4

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		audio_file = self.df.loc[idx, 'path']
		class_id = self.df.loc[idx, 'classID']

		aud = torchaudio.load(audio_file)

		reaud = self.resample(aud, self.sr)
		rechan = self.rechannel(reaud, self.channel)

		dur_aud = self.pad_trunc(rechan, self.duration)
		shift_aud = self.time_shift(dur_aud, self.shift_pct)
		sgram = self.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
		#img = Image.fromarray(sgram[1].numpy(), 'RGB')
		#img.save('my.png')
		#img.show()
		aug_sgram = self.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

		return aug_sgram, class_id

	def resample(self, aud, newsr):
		sig, sr = aud

		if (sr == newsr):
			return aud

		num_channels = sig.shape[0]
		resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
		if (num_channels > 1):
			retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
			resig = torch.cat([resig, retwo])

		return ((resig, newsr))

	def rechannel(self, aud, new_channel):
		sig, sr = aud

		if (sig.shape[0] == new_channel):
			return aud

		if (new_channel == 1):
			# Convert from stereo to mono by selecting only the first channel
			resig = sig[:1, :]
		else:
			# Convert from mono to stereo by duplicating the first channel
			resig = torch.cat([sig, sig])

		return ((resig, sr))

	def pad_trunc(self, aud, max_ms):
		sig, sr = aud
		num_rows, sig_len = sig.shape
		max_len = sr//1000 * max_ms

		if (sig_len > max_len):
			# Truncate the signal to the given length
			sig = sig[:,:max_len]

		elif (sig_len < max_len):
			# Length of padding to add at the beginning and end of the signal
			pad_begin_len = random.randint(0, max_len - sig_len)
			pad_end_len = max_len - sig_len - pad_begin_len
			# Pad with 0s
			pad_begin = torch.zeros((num_rows, pad_begin_len))
			pad_end = torch.zeros((num_rows, pad_end_len))

			sig = torch.cat((pad_begin, sig, pad_end), 1)

		return (sig, sr)

	def time_shift(self, aud, shift_limit):
		sig,sr = aud
		_, sig_len = sig.shape
		shift_amt = int(random.random() * shift_limit * sig_len)
		return (sig.roll(shift_amt), sr)

	def spectro_gram(self, aud, n_mels=64, n_fft=1024, hop_len=None):
		sig,sr = aud
		top_db = 80
		# spec has shape [channel, n_mels, time], where channel is mono, stereo etc
		spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
		# Convert to decibels
		spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

		return (spec)

	def spectro_augment(self, spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
		_, n_mels, n_steps = spec.shape
		mask_value = spec.mean()
		aug_spec = spec

		freq_mask_param = max_mask_pct * n_mels
		for _ in range(n_freq_masks):
			aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

		time_mask_param = max_mask_pct * n_steps
		for _ in range(n_time_masks):
			aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

		return aug_spec

class AudioClassifier (nn.Module):
	def __init__(self, nb_classes):
		super().__init__()
		conv_layers = []

		# First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
		self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
		self.relu1 = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(8)
		init.kaiming_normal_(self.conv1.weight, a=0.1)
		self.conv1.bias.data.zero_()
		conv_layers += [self.conv1, self.relu1, self.bn1]

		# Second Convolution Block
		self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu2 = nn.ReLU()
		self.bn2 = nn.BatchNorm2d(16)
		init.kaiming_normal_(self.conv2.weight, a=0.1)
		self.conv2.bias.data.zero_()
		conv_layers += [self.conv2, self.relu2, self.bn2]

		# Second Convolution Block
		self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu3 = nn.ReLU()
		self.bn3 = nn.BatchNorm2d(32)
		init.kaiming_normal_(self.conv3.weight, a=0.1)
		self.conv3.bias.data.zero_()
		conv_layers += [self.conv3, self.relu3, self.bn3]

		# Second Convolution Block
		self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		self.relu4 = nn.ReLU()
		self.bn4 = nn.BatchNorm2d(64)
		init.kaiming_normal_(self.conv4.weight, a=0.1)
		self.conv4.bias.data.zero_()
		conv_layers += [self.conv4, self.relu4, self.bn4]

		# Linear Classifier
		self.ap = nn.AdaptiveAvgPool2d(output_size=1)
		self.lin = nn.Linear(in_features=64, out_features=nb_classes)

		# Wrap the Convolutional Blocks
		self.conv = nn.Sequential(*conv_layers)

	def forward(self, x):
		# Run the convolutional blocks
		x = self.conv(x)

		# Adaptive pool and flatten for input to linear layer
		x = self.ap(x)
		x = x.view(x.shape[0], -1)

		# Linear layer
		x = self.lin(x)

		return x

	def train(self, model, ds, device, num_epochs):

		train_dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
		# Loss Function, Optimizer and Scheduler
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
		                                        	steps_per_epoch=int(len(train_dl)),
		                                        	epochs=num_epochs,
		                                        	anneal_strategy='linear')

		# Repeat for each epoch
		for epoch in range(num_epochs):
			running_loss = 0.0
			correct_prediction = 0
			total_prediction = 0

			# Repeat for each batch in the training set
			for i, data in enumerate(train_dl):
				# Get the input features and target labels, and put them on the GPU
				inputs, labels = data[0].to(device), data[1].to(device)

				# Normalize the inputs
				inputs_m, inputs_s = inputs.mean(), inputs.std()
				inputs = (inputs - inputs_m) / inputs_s

				# Zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
				scheduler.step()

				# Keep stats for Loss and Accuracy
				running_loss += loss.item()

				# Get the predicted class with the highest score
				_, prediction = torch.max(outputs,1)
				# Count of predictions that matched the target label
				correct_prediction += (prediction == labels).sum().item()
				total_prediction += prediction.shape[0]

			# Print stats at the end of the epoch
			num_batches = len(train_dl)
			avg_loss = running_loss / num_batches
			acc = correct_prediction/total_prediction
			print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
		torch.save(model, "./mnhn_model.pth" )
		print('Finished Training')

def main(argv):

	csvFile = "./dataset/training/config.csv"

	try:
		opts, args = getopt.getopt(argv[1:],"hc:",["config="])
	except getopt.GetoptError:
		print("{} -c <csv file>".format(argv[0]))
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("{} -c <csv file>".format(argv[0]))
			sys.exit()
		elif opt in ("-c", "--config"):
			csvFile = arg

	ds = SoundDataSet(csvFile)

	model = AudioClassifier(len(ds.classes))
	device = torch.device("cpu")
	if torch.cuda.is_available():
		device = torch.device("cuda")
		print("Using Cuda");
	model = model.to(device)
	# Check that it is on Cuda
	next(model.parameters()).device

	model.train(model, ds, device, num_epochs=1000)


if __name__ == "__main__":
   main(sys.argv)
