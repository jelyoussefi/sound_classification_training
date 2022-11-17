#!/usr/bin/python

import os, sys, getopt
import time
from datetime import datetime

import torch
from torch import nn
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

def train(model, ds, device, num_epochs, threshold=0.99, output_dir="./output"):

	train_dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
	# Loss Function, Optimizer and Scheduler
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
	                                        	steps_per_epoch=int(len(train_dl)),
	                                        	epochs=num_epochs,
	                                        	anneal_strategy='linear')

	writer = SummaryWriter()
	last_saved_acc = 0

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
			#scheduler.step()

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
		i_acc = int((acc)*100)
		if i_acc >= 10 and i_acc % 10 == 0 and last_saved_acc != i_acc:
			save(model,acc,output_dir)
			last_saved_acc = i_acc

		print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
		if acc >= threshold :
			save(model,acc,output_dir)
			break;

	writer.close()

	print('Finished Training')

def main(argv):

	csvFile = "./dataset/training/config.csv"
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
	
	ds = SoundDataSet(csvFile, device, ratio=0.01).to(device)
	model = CNNAudioClassifier(len(ds.classes)).to(device)

	train(model, ds, device, num_epochs=1000, output_dir=output_dir)


if __name__ == "__main__":
   main(sys.argv)
