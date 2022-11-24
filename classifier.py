#!/usr/bin/python

import os, sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
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
from IPython.display import Audio, display
import matplotlib.pyplot as plt
import librosa
from torchsummary  import summary

class AudioProcessor(nn.Module) :
	def __init__(self, device, duration, frame_rate):
		super().__init__()
		self.device = device
		self.duration = duration
		self.frame_rate = frame_rate
		self.shift_pct = 0.4		
	

	def get_spectrum(self, audio_file, start_time, duration, freq_peak=None, augment=False):
		
		info = torchaudio.info(audio_file)
		f_start_time = (start_time*info.sample_rate)/1000.0
		f_duration = (duration*info.sample_rate)/1000.0
		f_center = f_start_time + f_duration/2;
		f_audio_duration = int((self.duration*info.sample_rate)/1000.0)
		f_offset = max(0, int(f_center - f_audio_duration/2))
		sig, sr = torchaudio.load(audio_file, frame_offset=f_offset, num_frames=f_audio_duration)
		sig = sig.to(self.device)
		waveform = (sig, sr)
		if augment is True:
			waveform = self.time_shift(waveform, self.shift_pct)

		waveform = self.resample(waveform, self.frame_rate)
		waveform = self.down(waveform)
		waveform = self.pad_trunc(waveform, self.duration)
		
		sgram = self.spectro_gram(waveform,freq_peak)
		if augment is True:
			sgram = self.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
		return sgram
		

	def resample(self, waveform, newsr):
		sig, sr = waveform

		if (sr != newsr):
			sig = torchaudio.transforms.Resample(sr, newsr).to(self.device)(sig)			

		return (sig, newsr)

	def down(self, waveform):
		sig, sr = waveform

		if sig.shape[0] > 1:
			sig = torch.mean(sig, dim=0, keepdim=True)

		return (sig, sr)

	def pad_trunc(self, waveform, max_ms):
		sig, sr = waveform
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
			pad_begin = torch.zeros((num_rows, pad_begin_len)).to(self.device)
			pad_end = torch.zeros((num_rows, pad_end_len)).to(self.device)

			sig = torch.cat((pad_begin, sig, pad_end), 1)

		return (sig, sr)

	def time_shift(self, waveform, shift_limit):
		sig,sr = waveform
		_, sig_len = sig.shape
		shift_amt = int(random.random() * shift_limit * sig_len)
		return (sig.roll(shift_amt), sr)

	def spectro_gram(self, waveform, freq_peak=None):
		sig,sr = waveform
                
		spec = transforms.MelSpectrogram(sample_rate=self.frame_rate, n_fft=1024, hop_length=512, n_mels=64).to(self.device)(sig)

		spec = transforms.AmplitudeToDB(top_db=80).to(self.device)(spec)

		return spec

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

	def plot_waveform(self, waveform, title="Waveform", xlim=None, ylim=None):
		signal,sr = waveform
		signal = signal.numpy()
		num_channels, num_frames = signal.shape
		time_axis = torch.arange(0, num_frames) / sr
		figure, axes = plt.subplots(num_channels, 1)
		if num_channels == 1:
			axes = [axes]
		for c in range(num_channels):
			axes[c].plot(time_axis, signal[c], linewidth=1)
			axes[c].grid(True)
			if num_channels > 1:
				axes[c].set_ylabel(f'Channel {c+1}')
			if xlim:
				axes[c].set_xlim(xlim)
			if ylim:
				axes[c].set_ylim(ylim)
			figure.suptitle(title)
			plt.show(block=True)

	def plot_specgram(self, specgram, ylabel="freq_bin"):
		fig, axs = plt.subplots(1, 1)
		axs.set_title("Spectrogram (db)")
		axs.set_ylabel(ylabel)
		axs.set_xlabel("frame")
		im = axs.imshow(librosa.power_to_db(specgram[0]), origin="lower", aspect="auto")
		fig.colorbar(im, ax=axs)
		plt.show(block=True)

	def play_audio(self, waveform, sample_rate):
		#waveform = waveform.numpy()
		num_channels, num_frames = waveform.shape
		if num_channels == 1:
			display(Audio(waveform[0], rate=sample_rate))
		elif num_channels == 2:
			display(Audio((waveform[0], waveform[1]), rate=sample_rate))
		else:
			raise ValueError("Waveform with more than 2 channels are not supported.")



class SoundDataSet(AudioProcessor) :
	def __init__(self, metadata_file, device, labels_file=None, max_value=None):
		super().__init__(device, duration=1000, frame_rate=44100)
		
		self.df = pd.read_csv(metadata_file, sep=";", names=["path", "start_time", "duration", "frequency_peak"])
		self.df.head()

		ds_path = os.path.dirname(os.path.abspath(metadata_file))
		self.df['classID'] = np.array([c.split('-')[0] for c in self.df['path']])
		
		if max_value is not None:
			self.df =  self.df.groupby("classID").filter(lambda x: len(x) >= max_value)
			self.df = self.df.groupby("classID").sample(n=max_value, replace=False, random_state=1)
			self.df = self.df.sample(frac=1, ignore_index=True)


		if labels_file is None:
			self.classes = np.unique(self.df['classID'])

			with open(r'./classes.txt', 'w') as fp:
				fp.write('\n'.join(self.classes))
		else:
			with open(labels_file) as f:
				self.classes = np.array(f.read().splitlines())
				ds_classes = np.unique(self.df['classID'])
				diff = list(set(ds_classes) - set(self.classes))
				for cl in diff:
					self.df = self.df.loc[self.df['classID'] != cl ]

		self.df['classID'] = np.array([np.where(self.classes==c)[0] for c in self.df['classID']], dtype=object)
		self.df['path'] = ds_path + '/' + self.df['path'];
		
		
		#np.set_printoptions(linewidth=2000) 
		#print(np.array(self.df['classID'].value_counts()))

		print("\nCreating {} audio spectrums ... ".format(len(self.df)));
		sgrams = []
		st = time.time()
		for idx in range(len(self.df)):
			audio_file = self.df.loc[idx, 'path']
			start_time = self.df.loc[idx, 'start_time']
			duration   = self.df.loc[idx, 'duration']
			maxfreq = self.df.loc[idx, 'frequency_peak']
			sgram = self.get_spectrum(audio_file, start_time, duration, maxfreq, True)
			sgrams.append(sgram.cpu())

		print("Done in {} seconds".format(int(time.time() - st)))
		self.df['sgram'] = sgrams
		

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		class_id = self.df.loc[idx, 'classID']
		sgram = self.df.loc[idx, 'sgram']
		return sgram, class_id

	
class AudioClassifier (nn.Module):
	def __init__(self, nb_classes):
		super().__init__()
		
		self.conv1 = nn.Sequential (
			nn.Conv2d(1, 8, kernel_size=5),
			nn.ReLU(),
			nn.BatchNorm2d(8)
		)

		self.conv2 = nn.Sequential (
			nn.Conv2d(8, 16, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm2d(16)
		)

		self.conv3 = nn.Sequential (
			nn.Conv2d(16, 32, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm2d(32)
		)

		self.conv4 = nn.Sequential (
			nn.Conv2d(32, 64, kernel_size=3),
			nn.ReLU(),
			nn.BatchNorm2d(64)
		)

		# Linear Classifier
		self.ap = nn.AdaptiveAvgPool2d(output_size=1)
		self.lin = nn.Linear(in_features=64, out_features=nb_classes)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)

		# Adaptive pool and flatten for input to linear layer
		x = self.ap(x)
		x = x.view(x.shape[0], -1)

		# Linear layer
		x = self.lin(x)

		return x
class CNNAudioClassifier(nn.Module):
    # ----------------------------
    # Introduce the new model architecture
    # ----------------------------
    def __init__(self,nb_classes):
        super().__init__()

        # 4 CNN block / flatten / linear / softmax
        self.conv1 = nn.Sequential(
                     nn.Conv2d( in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2 ),
                     nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.conv2 = nn.Sequential(
                     nn.Conv2d( in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2 ),
                     nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.conv3 = nn.Sequential(
                     nn.Conv2d( in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2 ),
                     nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.conv4 = nn.Sequential( nn.Conv2d( in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2 ),
                     nn.ReLU(), nn.MaxPool2d(kernel_size=2) )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128*5*7, out_features=nb_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__== "__main__":
	cnn = CNNAudioClassifier(49)
	summary(cnn, (1,64,86))
