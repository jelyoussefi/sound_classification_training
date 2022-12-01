#!/usr/bin/python

import os, sys, getopt
import time
from datetime import datetime

import numpy as n
import torch
from torchvision import transforms
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions
from audio_processor import AudioProcessor
import torchvision.transforms as T
from torchvision.models import resnet50

def preprocess(img):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ]
    )
    return transform(img)
    
def infer(device, model, input_file):
	ap = AudioProcessor(device, 1000, 41100)
	image = ap.get_spectrum(input_file, start_time=2224, duration=738)
	#image_m, image_s = (image*0.1).mean(), (image*0.1).std()
	#image = 255.0*((image - image_m) / image_s)
	transform = T.ToPILImage()
	image = transform(image)
	image = preprocess(image)
	image = torch.unsqueeze(image, 0)
	outputs=model(image)
	probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
	top5_prob, top5_catid = torch.topk(probabilities, 5)
	print("Top 5 Results: \nLabels , Probabilities:")
	for i in range(top5_prob.size(0)):
		print(top5_catid[i], top5_prob[i].item())


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
	
	model = torch.load(model_path).to(device)
	print(model)
	provider_options = OpenVINOProviderOptions(backend = "CPU", precision = "FP32")
	model = ORTInferenceModule(model, provider_options = provider_options)
	model.eval()


	infer(device, model, input_file)

if __name__ == "__main__":
   main(sys.argv)
