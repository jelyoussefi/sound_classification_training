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
from torchvision.models import resnet34
from torch.utils.tensorboard import SummaryWriter
from classifier import SoundDataSet, CNNAudioClassifier


def save(model, output_dir):
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

	csv_file = "./dataset/training/config.csv"
	output_dir = "./model"

	try:
		opts, args = getopt.getopt(argv[1:],"hc:o:",["config=","ouput_dir="])
	except getopt.GetoptError:
		print("{} -c <csv file>".format(argv[0]))
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("{} -c <csv file>".format(argv[0]))
			sys.exit()
		elif opt in ("-c", "--config"):
			csv_file = arg
		elif opt in ("-o", "--ouput_dir"):
			output_dir = arg

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device : ", device)

	ds = SoundDataSet(csv_file, device, max_value=1000).to(device)
	print(len(ds.classes))
	num_items = len(ds)
	num_train = round(num_items * 0.8)
	num_val = num_items - num_train
	train_ds, valid_ds = random_split(ds, [num_train, num_val])

	model = resnet34(pretrained=True) #weights=ResNet34_Weights.DEFAULT
	model.fc = nn.Linear(512,len(ds.classes))
	model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
	model = model.to(device)

	#----------------------------------------------------------------------------------------
	lr = 0.001
	epochs = 50
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=16, shuffle=True)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()
	#classes = np.array([np.where(train_ds.classes==c)[0] for c in train_ds.classes], dtype=object)
	writer = SummaryWriter()

	for epoch in range(epochs):
		print(f"Epoch {epoch+1} of {epochs}")
		train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
		valid_loss, valid_acc = validate(model, device, valid_loader, criterion, ds.classes)
		print(f'\tAccuracy\t train : {train_acc:.2f}, valid: {valid_acc:.2f}')

		writer.add_scalar("Acc/train", train_acc, epoch)
		writer.add_scalar("Acc/valid", valid_acc, epoch)

		writer.flush()

		if train_acc > 99:
			break;

	save(model,output_dir)
	writer.close()
	print('Finished Training')

if __name__ == "__main__":
   main(sys.argv)
