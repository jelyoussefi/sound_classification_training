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
    
def infer(device, model, input_file):
	ap = AudioProcessor(device, 1000, 44100)
	image = ap.audio_to_image(input_file, start_time=556, duration=150, resize=True) #class ID 10,sr=44100,Otusco--Otusco--Otus-scops-579749-R9.wav;28;470;1378
											#class ID 0, sr=44100, Alaarv--Alaarv--Alauda-arvensis-52384-R1.wav;556;150;4995
	#image = preprocess(image)
	image = torch.unsqueeze(image, 0)
	outputs=model(image)
	print(outputs)
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
	
	classes = []
	with open("./model/labels.txt") as f:
		classes = np.array(f.read().splitlines())

	model = AudioCNN(len(classes))()
	model.load_state_dict(torch.load(model_path))

	provider_options = OpenVINOProviderOptions(backend = "CPU", precision = "FP32")
	#model = ORTInferenceModule(model, provider_options = provider_options)
	model.eval()


	infer(device, model, input_file)

if __name__ == "__main__":
   main(sys.argv)
