#!/usr/bin/python

import os, sys, getopt
import time
from datetime import datetime

import numpy as np
import torch
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions
from audio_processor import AudioProcessor


def infer(device, model, input_file):
	print(input_file)
	ap = AudioProcessor(device, 1000, 41100)
	spec = ap.get_spectrum(input_file, start_time=1158, duration=738)
	print(spec.shape)
	output=model(spec)
	print(output)

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
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Device : ", device)

	model = torch.load(model_path).to(device)
	provider_options = OpenVINOProviderOptions(backend = "CPU", precision = "FP32")
	model = ORTInferenceModule(model, provider_options = provider_options)
	model.eval()


	infer(device, model, input_file)

if __name__ == "__main__":
   main(sys.argv)
