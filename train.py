#!/usr/bin/python

import os, sys, getopt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import time

from classifier import SoundDataSet, AudioClassifier


def train(model, ds, device, num_epochs):

	train_dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
	# Loss Function, Optimizer and Scheduler
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
	                                        	steps_per_epoch=int(len(train_dl)),
	                                        	epochs=num_epochs,
	                                        	anneal_strategy='linear')

	writer = SummaryWriter()

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
		writer.add_scalar("Acc/train", acc, epoch)
		writer.flush()

		print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
	torch.save(model, "./mnhn_model.pth" )
	writer.close()

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

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device : ", device)
	
	ds = SoundDataSet(csvFile, device)

	model = AudioClassifier(len(ds.classes))
	model.to(device)

	# Check that it is on Cuda
	#next(model.parameters()).device

	train(model, ds, device, num_epochs=1000)


if __name__ == "__main__":
   main(sys.argv)
