#!/usr/bin/python

import os, sys, getopt
import time

import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from classifier import SoundDataSet, AudioClassifier


class Inference :
	def __init__(self, device, model):
		self.device = device
		self.model = model
		
	def run(self, ds):
		correct_prediction = 0
		total_prediction = 0
		val_dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
		# Disable gradient updates
		with torch.no_grad():
			for data in val_dl:
				# Get the input features and target labels, and put them on the GPU
				inputs, labels = data[0].to(self.device), data[1].to(self.device)

				# Normalize the inputs
				inputs_m, inputs_s = inputs.mean(), inputs.std()
				inputs = (inputs - inputs_m) / inputs_s

				# Get predictions
				outputs = self.model(inputs)
				print(outputs)
				# Get the predicted class with the highest score
				_, prediction = torch.max(outputs,1)
				# Count of predictions that matched the target label
				correct_prediction += (prediction == labels).sum().item()
				total_prediction += prediction.shape[0]

				acc = correct_prediction/total_prediction
				print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

def main(argv):

	csvFile = "./dataset/validation/config.csv"
	model_path = None
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
	
	model = torch.load(model_path)

	ds = SoundDataSet(csvFile, device, labels_file="./classes.txt", ratio=0.5)
	inf = Inference(device, model)

	inf.run(ds)


if __name__ == "__main__":
   main(sys.argv)
