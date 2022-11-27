#!/usr/bin/python

import os, sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import pandas as pd
import numpy as np
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
from torchvision.models import resnet34
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

	def spectro_to_image(self, spec, eps=1e-6):
		mean = spec.mean()
		std = spec.std()
		spec_norm = (spec - mean) / (std + eps)
		spec_min, spec_max = spec_norm.min(), spec_norm.max()
		spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
		#spec_scaled = spec_scaled.type(torch.uint8)
		return spec_scaled

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
	def __init__(self, device, metadata_file=None, df=None, duration=1000, min_number=None, max_number=None):
		super().__init__(device, duration=duration, frame_rate=44100)
		
		if df is not None:
			self.df = df 
			return

		self.df = pd.read_csv(metadata_file, sep=";", names=["path", "start_time", "duration", "frequency_peak"])
		self.df.head()	

		self.df = self.df[self.df.duration >= 50]
		self.df = self.df[self.df.duration <= 4000]

		#print(self.df)		

		self.df.insert(loc=len(self.df.columns), column='sgram', value=object);
		
		ds_path = os.path.dirname(os.path.abspath(metadata_file))
		self.df['class_id'] = np.array([c.split('-')[0] for c in self.df['path']])
		
		self.df = self.df.sort_values(by=['duration'], ascending=False)
		self.df = self.df[self.df["duration"] >= 116]
		
		#self.df =  self.df[self.df.duration <= 5000]
		
		if min_number is not None:
			self.df = self.df.sort_values(by=['duration'], ascending=False)
			self.df =  self.df.groupby("class_id").filter(lambda x: len(x) >= min_number)
			self.df = self.df.sample(frac=1, ignore_index=True)
		
		if max_number is not None:
			self.df = self.df.sort_values(by=['duration'], ascending=False)
			self.df = self.df.groupby("class_id").head(max_number)
			self.df = self.df.sample(frac=1, ignore_index=True)

		self.classes = np.unique(self.df['class_id'])

		self.df = self.df.sample(frac=1, ignore_index=True)
		self.df['class_id'] = np.array([np.where(self.classes==c)[0] for c in self.df['class_id']], dtype=object)
		self.df['path'] = ds_path + '/' + self.df['path'];
		
		np.set_printoptions(linewidth=2000) 
		print(np.array(self.df['class_id'].value_counts()))
		print(np.array(self.df['duration']))

		print("\nCreating {} audio spectrums ... ".format(len(self.df)));
		st = time.time()
		ds_len = len(self.df);

		for idx in range(len(self.df)):
			audio_file = self.df.loc[idx, 'path']
			start_time = self.df.loc[idx, 'start_time']
			duration   = self.df.loc[idx, 'duration']
			freq_peak = self.df.loc[idx, 'frequency_peak']
			class_id = self.df.loc[idx, 'class_id']
			sgram = self.get_spectrum(audio_file, start_time, duration, freq_peak, False)
			self.df.loc[idx, 'sgram'] = sgram

			sgram = self.get_spectrum(audio_file, start_time, duration, freq_peak, True)
			aug_row = [ audio_file, start_time, duration, freq_peak, sgram, class_id ]
			
			self.df = self.df.append(pd.Series(aug_row, index=self.df.columns[:len(aug_row)]), ignore_index=True)

		print("Done in {} seconds".format(int(time.time() - st)))
				
		
	def split(self, train_ratio=0.8):

		train_df = self.df.groupby('class_id').sample(frac=train_ratio, random_state=1)
		valid_df = self.df.drop(train_df.index)
		train_df = train_df.sample(frac=1, ignore_index=True)
		valid_df = valid_df.sample(frac=1, ignore_index=True)

		train_ds = SoundDataSet(self.device, df=train_df)
		sig, cid = train_ds[0]
		valid_ds = SoundDataSet(self.device, df=valid_df)

		return train_ds,valid_ds


	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		class_id = self.df.loc[idx, 'class_id']
		sgram = self.df.loc[idx, 'sgram']
		return sgram, class_id


def AudioCNN(nb_classes):
	model = resnet34(pretrained=True) #weights=ResNet34_Weights.DEFAULT
	model.fc = nn.Linear(512,nb_classes)
	model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
	return model

