#!/usr/bin/python

import os, sys, getopt
import time
import torchaudio
import math, random

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from classifier import SoundDataSet

def resample(waveform, newsr):
	sig, sr = waveform

	if (sr != newsr):
		sample = torchaudio.transforms.Resample(sr, newsr)	
		sig = sample(sig)	

	return (sig, newsr)

def down(waveform):
	sig, sr = waveform

	if sig.shape[0] > 1:
		sig = torch.mean(sig, dim=0, keepdim=True)

	return (sig, sr)

def pad_trunc(waveform, max_ms):
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
		pad_begin = torch.zeros((num_rows, pad_begin_len))
		pad_end = torch.zeros((num_rows, pad_end_len))

		sig = torch.cat((pad_begin, sig, pad_end), 1)

	return (sig, sr)

def random_concatenate(metadata_file,labels_file,cat_number,output_dir): 
		
	fix_sample_duration=1000	
	fix_frame_rate=44100	
	df = pd.read_csv(metadata_file, sep=";", names=["path", "start_time", "duration", "frequency_peak"])
	df.head()	

	ds_path = os.path.dirname(os.path.abspath(metadata_file))
	df['class'] = np.array([c.split('-')[0] for c in df['path']])
	print(np.array(df['class'].value_counts()))

	if labels_file is not None:
		with open(labels_file) as f:
			classes = np.array(f.read().splitlines())
	print(classes)
	ds_classes = np.unique(df['class'])
	diff = list(set(ds_classes) - set(classes))
	for cl in diff:
		df = df.loc[df['class'] != cl ]
	else:
		classes = np.unique(df['class'])

	df = df.sample(frac=1, ignore_index=True)
	df['class_id'] = np.array([np.where(classes==c)[0] for c in df['class']], dtype=object)
	df['path'] = ds_path + '/' + df['path'];
		
	np.set_printoptions(linewidth=2000) 
	#print(np.array(df['class'].value_counts()))

	print("\nCreating {} audio spectrums ... ".format(len(df)));
	st = time.time()
	ds_len = len(df);

	for idx in range(ds_len-cat_number):
		output_audio=torch.zeros(0)
		output_filename="" 
		for index in range(cat_number):
			audio_file = df.loc[idx+index, 'path']
			start_time = df.loc[idx+index, 'start_time']
			duration   = df.loc[idx+index, 'duration']
			class_ = df.loc[idx+index, 'class']
			info = torchaudio.info(audio_file)
			f_start_time = (start_time*info.sample_rate)/1000.0
			f_duration = (duration*info.sample_rate)/1000.0
			f_center = f_start_time + f_duration/2;
			f_audio_duration = int((fix_sample_duration*info.sample_rate)/1000.0)
			f_offset = max(0, int(f_center - f_audio_duration/2))
			if index !=0 : 
				output_filename=output_filename+"--"+str(class_)+"-"+str(duration) 
			else : 
				output_filename=output_filename+str(class_)+"-"+str(duration) 

			sig, sr = torchaudio.load(audio_file, frame_offset=f_offset, num_frames=f_audio_duration)
			waveform = (sig, sr)

			waveform = resample(waveform, fix_frame_rate)
			waveform = down(waveform)
			waveform = pad_trunc(waveform, fix_sample_duration)
			output_audio=torch.cat((output_audio,waveform[0]),1)
		file_path=os.path.join(output_dir,output_filename+".wav") 
		torchaudio.save(file_path,output_audio,fix_frame_rate)
		

def main(argv):

	csv_file    = "./dataset/validation/config.csv"
	labels_path = "./labels.txt"
	output_dir = "./"
	cat_number=5
	try:
		opts, args = getopt.getopt(argv[1:],"hc:m:l:",["config=", "num=",  "labels="])
	except getopt.GetoptError:
		print("{} -c <csv file> -n <num>".format(argv[0]))
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("{} -c <csv file> -n <num> -l <labels>".format(argv[0]))
			sys.exit()
		elif opt in ("-c", "--config"):
			csvFile = arg
		elif opt in ("-n", "--num"):
			model_path = arg
		elif opt in ("-l", "--labels"):
			labesl_path = arg
		elif opt in ("-o", "--ouput_dir"):
            		output_dir = arg

	if model_path is None:
		print("{} -c <csv file> -m <model> -l <labels> ".format(argv[0]))
		sys.exit(2)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device : ",device)
	
	random_concatenate(csv_file,labels_path,cat_number,output_dir)

if __name__ == "__main__":
   main(sys.argv)
