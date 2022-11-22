#!/usr/bin/python

import random
import math
from torchsummary import summary
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from torchaudio import transforms
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torchaudio
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import time
import os
import sys
import getopt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"



def SoundDataSet(output_dir, metadata_file, num_classes=18, num_files=500, labels_file=None, ratio=1):
        number_of_classes=num_classes
        number_of_files=num_files
        number_of_rows=number_of_classes*number_of_files
        csv_path=os.path.join(output_dir,"config_"+str(num_classes)+"class_"+str(num_files)+"files"+".csv")
        print(csv_path)
        st = time.time()

        ##-------Read the Total data base,bigger CSV file-----------#
        df = pd.read_csv(metadata_file, sep=";", names=[
                              "path", "start_time", "duration", "frequency_peak"])
        df.head()

        ds_path = os.path.dirname(os.path.abspath(metadata_file))
        df['classID'] = np.array(
            [c.split('--')[0] for c in df['path']])

        if labels_file is None:
            classes = np.unique(df['classID'])

            with open(r'./classes.txt', 'w') as fp:
                fp.write('\n'.join(classes))
        else:
            with open(labels_file) as f:
                classes = np.array(f.read().splitlines())
                print(classes)
        ##-------- Display the file and classes read in the bigger data set ----------------- 

        total_filecount = df['classID'].value_counts()
        print(total_filecount)

        ##-------- Filter the files we need ----------------- 
        skip_rows=set(range(0,0))
        offset=0
        class_filter=number_of_files
        for i in range(len(classes)):
            if total_filecount[classes[i]] > class_filter:
               skip_rows.update(set(range(class_filter+offset,offset+total_filecount[classes[i]])))
            else:
               skip_rows.update(set(range(offset,offset+total_filecount[classes[i]])))
            offset=offset+total_filecount[classes[i]]

        Newdf = pd.read_csv(metadata_file, skiprows=skip_rows, nrows=number_of_rows-1)
        print(Newdf.count())
        Newdf.to_csv(csv_path, index=False)

        print("\nCreating {} audio dataset configuration ... ".format(len(Newdf)))
        print("Done in {} seconds".format(int(time.time() - st)))



def main(argv):

    csvFile = "../../v1/sound_classification_training/dataset/training/config.csv"
    output_dir = "./"
    num_files=1000
    num_classes=15

    try:
        opts, args = getopt.getopt(
            argv[1:], "hc:o:", ["config=", "ouput_dir="])
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

    SoundDataSet(output_dir,csvFile, num_classes,num_files, ratio=1)



if __name__ == "__main__":
    main(sys.argv)
