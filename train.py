#!/usr/bin/python

import os, sys, getopt
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet34
from torch.utils.tensorboard import SummaryWriter
from classifier import SoundDataSet, CNNAudioClassifier


def save(model, acc, output_dir):
	acc = int(acc*100)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	date_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
	model_path=os.path.join(output_dir,"model_"+date_time+"_acc-"+str(acc)+".pth")
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

def train(model, device, loss_fn, train_ds, valid_ds, epochs, optimizer, train_losses, valid_losses, change_lr=None):
	
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
	valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=16, shuffle=True)

	for epoch in tqdm(range(1,epochs+1)):
		model.train()
		batch_losses=[]
		if change_lr:
			optimizer = change_lr(optimizer, epoch)

		for i, data in enumerate(train_loader):
			x, y = data
			optimizer.zero_grad()
			x = x.to(device, dtype=torch.float32)
			y = y.to(device, dtype=torch.long)
			y_hat = model(x)
			loss = loss_fn(y_hat, y)
			loss.backward()
			batch_losses.append(loss.item())
			optimizer.step()
		train_losses.append(batch_losses)
		print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1]):.2f}')
		model.eval()
		batch_losses=[]
		trace_y = []
		trace_yhat = []

		for i, data in enumerate(valid_loader):
			x, y = data
			x = x.to(device, dtype=torch.float32)
			y = y.to(device, dtype=torch.long)
			y_hat = model(x)
			loss = loss_fn(y_hat, y)
			trace_y.append(y.cpu().detach().numpy())
			trace_yhat.append(y_hat.cpu().detach().numpy())      
			batch_losses.append(loss.item())
		valid_losses.append(batch_losses)
		trace_y = np.concatenate(trace_y)
		trace_yhat = np.concatenate(trace_yhat)
		accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
		print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1]):.2f} Valid-Accuracy : {accuracy:.2f}')


def main(argv):

	train_csv_file = "./dataset/training/config.csv"
	valid_csv_file = "./dataset/validation/config.csv"

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
			csvFile = arg
		elif opt in ("-o", "--ouput_dir"):
			output_dir = arg

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device : ", device)

	train_ds = SoundDataSet(train_csv_file, device, max_value=2000).to(device)
	valid_ds = SoundDataSet(valid_csv_file, device, labels_file="./classes.txt",  max_value=200).to(device)

	model = resnet34(pretrained=True) #weights=ResNet34_Weights.DEFAULT
	model.fc = nn.Linear(512,len(train_ds.classes))
	model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
	model = model.to(device)

	#----------------------------------------------------------------------------------------
	learning_rate = 2e-4
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	loss_fn = nn.CrossEntropyLoss()
	epochs = 50
	resnet_train_losses=[]
	resnet_valid_losses=[]

	#----------------------------------------------------------------------------------------
	
	train(model, device, loss_fn, train_ds, valid_ds, epochs, optimizer, resnet_train_losses, resnet_valid_losses, lr_decay)


if __name__ == "__main__":
   main(sys.argv)
