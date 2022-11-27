#!/usr/bin/python

import os, sys, getopt
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from classifier import SoundDataSet


def validate(model, device, valid_loader, loss_fn, class_names):
	model.eval()
	valid_running_loss = 0.0
	valid_running_correct = 0
	counter = 0

	# we need two lists to keep track of class-wise accuracy
	class_correct = list(0. for i in range(len(class_names)))
	class_total = list(0. for i in range(len(class_names)))

	with torch.no_grad():
		for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
			counter += 1
            
			image, labels = data
			image = image.to(device)
			labels = labels.to(device)
			# forward pass
			outputs = model(image)
			# calculate the loss
			loss = loss_fn(outputs, labels)
			valid_running_loss += loss.item()
			# calculate the accuracy
			_, preds = torch.max(outputs.data, 1)
			valid_running_correct += (preds == labels).sum().item()

			# calculate the accuracy for each class
			correct  = (preds == labels).squeeze()
			for i in range(len(preds)):
				label = labels[i]
				class_correct[label] += correct[i].item()
				class_total[label] += 1
        
	# loss and accuracy for the complete epoch
	epoch_loss = valid_running_loss / counter
	epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))

	print('-----------------------------------------------------------------------------')
	print('                            Accuracy per class')
	print('-----------------------------------------------------------------------------')
	for i in range(len(class_names)):
		if class_total[i] != 0:
			print(f"\t{i}\t{class_names[i]}:\t\t{100*class_correct[i]/class_total[i]:04.1f}")
	print('-----------------------------------------------------------------------------')
        
	return epoch_loss, epoch_acc


def main(argv):

	csvFile = "./dataset/validation/config_l.csv"
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

	ds = SoundDataSet(csvFile, device, labels_file="./classes.txt", ratio=1)
	inf = Inference(device, model)

	inf.run(ds)


if __name__ == "__main__":
   main(sys.argv)
