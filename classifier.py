#!/usr/bin/python

import os, sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import pandas as pd
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet34
from audio_processor import AudioProcessor
import warnings

warnings.filterwarnings("ignore")


class SoundDataSet(AudioProcessor) :
	def __init__(self, device, metadata_file=None, classes=None, df=None, duration=1000, min_number=None, max_number=None):
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
		self.df['class'] = np.array([c.split('-')[0] for c in self.df['path']])
		
		self.df = self.df.sort_values(by=['duration'], ascending=False)
				
		if min_number is not None:
			self.df = self.df.sort_values(by=['duration'], ascending=False)
			self.df =  self.df.groupby("class").filter(lambda x: len(x) >= min_number)
			self.df = self.df.sample(frac=1, ignore_index=True)
		
		if max_number is not None:
			self.df = self.df.sort_values(by=['duration'], ascending=False)
			self.df = self.df.groupby("class").head(max_number)
			self.df = self.df.sample(frac=1, ignore_index=True)

		if classes is not None:
			self.classes = classes
			ds_classes = np.unique(self.df['class'])
			diff = list(set(ds_classes) - set(self.classes))
			for cl in diff:
				self.df = self.df.loc[self.df['class'] != cl ]
		else:
			self.classes = np.unique(self.df['class'])

		self.df = self.df.sample(frac=1, ignore_index=True)
		self.df['class_id'] = np.array([np.where(self.classes==c)[0] for c in self.df['class']], dtype=object)
		self.df['path'] = ds_path + '/' + self.df['path'];
		
		np.set_printoptions(linewidth=2000) 
		print(np.array(self.df['class'].value_counts()))

		print("\nCreating {} audio spectrums ... ".format(len(self.df)));
		st = time.time()
		ds_len = len(self.df);

		for idx in range(len(self.df)):
			audio_file = self.df.loc[idx, 'path']
			start_time = self.df.loc[idx, 'start_time']
			duration   = self.df.loc[idx, 'duration']
			freq_peak = self.df.loc[idx, 'frequency_peak']
			class_ = self.df.loc[idx, 'class']
			class_id = self.df.loc[idx, 'class_id']
			sgram = self.get_spectrum(audio_file, start_time, duration, freq_peak, False)
			self.df.loc[idx, 'sgram'] = sgram.cpu()

			sgram = self.get_spectrum(audio_file, start_time, duration, freq_peak, True)
			aug_row = [ audio_file, start_time, duration, freq_peak, sgram.cpu(), class_, class_id ]
			
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

