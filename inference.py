#!/usr/bin/python

import os, sys, getopt
import time
import torchaudio
from datetime import datetime

import numpy as np
import torch
from torchvision import transforms
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions
from audio_processor import AudioProcessor
import torchvision.transforms as T
from classifier import AudioCNN

def preprocess(img):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
    			transforms.Resize(224),
    			transforms.ToTensor(),
    			transforms.Normalize(
        			mean=[0.485, 0.456, 0.406],
        			std=[0.229, 0.224, 0.225]
        		)
        ]
    )
    return transform(img)
    
def infer(device, model, input_file, classes):
	ap = AudioProcessor(device, 1000, 44100)
	start_time = 0
	while start_time < (5000-100):
		image = ap.audio_to_image(input_file, start_time=start_time, duration=1000, resize=True) 
		image = torch.unsqueeze(image, 0)
		outputs=model(image)
		probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
		top_prob, top_catid = torch.topk(probabilities, 1)
		print("---------------------------------------------------")
		print(f'\t{start_time}\t{classes[top_catid[0]]} : {top_prob[0].item():.2f}')
		start_time += 100


def main(argv):
	
	model_path = None
	input_file = None

	try:
		opts, args = getopt.getopt(argv[1:],"hm:i:",["model=","input="])
	except getopt.GetoptError:
		print("{} -m <model> -i <input>".format(argv[0]))
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("{} -m <model> -i <input>".format(argv[0]))
			sys.exit()
		elif opt in ("-m", "--model"):
			model_path = arg
		elif opt in ("-i", "--input"):
			input_file = arg
		
	if model_path is None or input_file is None:
		print("{} -m <model> -i <input>".format(argv[0]))
		sys.exit(2)
	device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device : ", device)
	
	classes = []
	with open("./model/labels.txt") as f:
		classes = np.array(f.read().splitlines())

	model = AudioCNN(len(classes))()
	model.load_state_dict(torch.load(model_path))

	provider_options = OpenVINOProviderOptions(backend = "CPU", precision = "FP32")
	model = ORTInferenceModule(model, provider_options = provider_options)
	model.eval()


	infer(device, model, input_file, classes)

if __name__ == "__main__":
   main(sys.argv)
