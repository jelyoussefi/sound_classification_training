#!/usr/bin/python

import os, sys, getopt
import time

import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from classifier import SoundDataSet, AudioCNN


class Inference :
	def __init__(self, device, model):
		self.device = device
		self.model = model
		
	def run(self, ds):
		correct_prediction = 0
		total_prediction = 0
		val_dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
		# Disable gradient updates
		with torch.no_grad():
			for data in val_dl:
				# Get the input features and target labels, and put them on the GPU
				inputs, labels = data[0].to(self.device), data[1].to(self.device)

				
				# Normalize the inputs
				#inputs_m, inputs_s = inputs.mean(), inputs.std()
				#inputs = (inputs - inputs_m) / inputs_s

				# Get predictions
				outputs = self.model(inputs)
				_, prediction = torch.max(outputs,1)
				# Get the predicted class with the highest score
				# Count of predictions that matched the target label
				correct_prediction += (prediction == labels).sum().item()
				total_prediction += prediction.shape[0]

				acc = correct_prediction/total_prediction
				print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

def main(argv):

	testcsv_file    = "./dataset/test/config.csv"
	model_path = None
	labels_file="./model/labels.txt"
	try:
		opts, args = getopt.getopt(argv[1:],"hc:m:",["config=", "model="])
	except getopt.GetoptError:
		print("{} -c <csv file> -m <model>".format(argv[0]))
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("{} -c <csv file> -m <model> ".format(argv[0]))
			sys.exit()
		elif opt in ("-c", "--config"):
			csvFile = arg
		elif opt in ("-m", "--model"):
			model_path = arg

	if model_path is None:
		print("{} -c <csv file> -m <model>".format(argv[0]))
		sys.exit(2)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device : ",device)
	
	with open(labels_file) as f:
		classes = np.array(f.read().splitlines())

	ds = SoundDataSet(device, testcsv_file, classes, duration=1000, min_number=10, max_number=2500)
	model = AudioCNN(len(ds.classes))().to(device)
	model.load_state_dict(torch.load(model_path))#model = torch.load(model_path)
	inf = Inference(device, model)
	print(f'Total files...... {len(ds)}')
	inf.run(ds)


if __name__ == "__main__":
   main(sys.argv)
