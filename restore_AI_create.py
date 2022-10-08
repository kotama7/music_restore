import os
import pickle
import wave
import glob
import torch
from torch import nn

directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(directory,"dataset") +"\*.wav"
files = glob.glob(dataset_path)
dataset = [] * len(files)
print(files,dataset_path)
for path, c in zip(files,range(len(files))):
    with wave.open(path,'rb') as f:
        dataset[c] = f


pickle.dump(restore_AI,f'{directory}/AI/restore_AI.sav')