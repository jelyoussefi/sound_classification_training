#!/usr/bin/python

import os, sys, getopt
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from classifier import SoundDataSet, AudioCNN
from valid import validate

def save(model, acc, output_dir):
	acc = int(acc*100)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	date_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
	model_path=os.path.join(output_dir,"model_"+date_time+".pth")
	print("\nSaving ", model_path, " model ...\n")
	torch.save(model, model_path)
	model_path_symblink = os.path.join(output_dir, "model.pth")
	if os.path.islink(model_path_symblink):
		os.remove(model_path_symblink)
	os.symlink(os.path.basename(model_path), model_path_symblink)

def lr_decay(optimizer, epoch):
	if epoch%10==0:
		new_lr = learning_rate / (10**(epoch//10))
		optimizer = setlr(optimizer, new_lr)
		print(f'Changed learning rate to {new_lr}')
	return optimizer


def train(model, device, train_loader, optimizer, loss_fn):

		model.train()
		train_running_loss = 0.0
		train_running_correct = 0
		counter = 0
		for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
			counter += 1
			image, labels = data
			optimizer.zero_grad()
			image = image.to(device, dtype=torch.float32)
			labels = labels.to(device, dtype=torch.long)
			
			# forward pass
			outputs = model(image)
			
			# calculate the loss
			loss = loss_fn(outputs, labels)
			train_running_loss += loss.item()
			
			 # calculate the accuracy
			_, preds = torch.max(outputs.data, 1)
			train_running_correct += (preds == labels).sum().item()
			
			# backpropagation			
			loss.backward()

			# update the optimizer parameters
			optimizer.step()

		# loss and accuracy for the complete epoch
		epoch_loss = train_running_loss / counter
		epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))

		return epoch_loss, epoch_acc 


def main(argv):
	csv_file = "./dataset/training/config.csv"
	output_dir = "./model"
	csv_valid_file = None

	try:
		opts, args = getopt.getopt(argv[1:],"hc:v:o:",["config=","config_valid=", "ouput_dir="])
	except getopt.GetoptError:
		print("{} -c <csv file>".format(argv[0]))
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("{} -c <csv file>".format(argv[0]))
			sys.exit()
		elif opt in ("-c", "--config"):
			csv_file = arg
		elif opt in ("-v", "--config_valid"):
			csv_valid_file = arg
		elif opt in ("-o", "--ouput_dir"):
			output_dir = arg

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device : ", device)

	ds = SoundDataSet(device, metadata_file=csv_file, duration=1000, min_number=100, max_number=200).to(device)
	if csv_valid_file is None:
		train_ds, valid_ds = ds.split(0.8)
	else:
		train_ds = ds;
		valid_ds = SoundDataSet(device, metadata_file=csv_valid_file, classes=ds.classes, duration=1000).to(device)

	model = AudioCNN(len(ds.classes)).to(device)

	with open(os.path.join(output_dir,"labels.txt"), 'w') as fp:
		fp.write('\n'.join(ds.classes))
	
	writer = SummaryWriter()

	#----------------------------------------------------------------------------------------
	lr = 0.001
	epochs = 50
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=16, shuffle=True)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(epochs):
		print(f"Epoch {epoch+1} of {epochs}")
		train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
		valid_loss, valid_acc = validate(model, device, valid_loader, criterion, ds.classes)
		print(f'\tAccuracy\t train : {train_acc:.2f}, valid: {valid_acc:.2f}')

		writer.add_scalars('Accuracy', {'train': train_acc}, epoch)
		writer.add_scalars('Accuracy', {'valid': valid_acc}, epoch)

		writer.flush()

		if train_acc > 99:
			break;

	save(model,train_acc,output_dir)
	writer.close()
	print('Finished Training')

if __name__ == "__main__":
   main(sys.argv)
