import os
import pickle
import wave
import glob
import torch
import re
from torch import nn
from torch.utils.data import DataLoader

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()
discriminator = Discriminator()
lr = 0.001  #prtotype
num_epochs = 300    #prtotype
loss_function = nn.BCELoss()    #prtotype
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
torch.manual_seed(111)
directory = os.path.dirname(os.path.abspath(__file__))
x_dataset_path = os.path.join(directory,"x_dataset") +"\*.wav"
y_dataset_path = os.path.join(directory,'y_dataset') + "\*.wav"
x_files = glob.glob(x_dataset_path)
y_files = glob.glob(y_dataset_path)
x_files.sort()
y_files.sort()
dataset_len = len(x_files)
if x_files != y_files:
    re.error("dataset is not paired by x and y")
dataset = [() * 2] * dataset_len
for path, c in zip(x_files,range(dataset_len)):
    with wave.open(path,'rb') as f:
        dataset[c][0] = f
for path, c in zip(y_files,range(dataset_len)):
    with wave.open(path,'rb') as f:
        dataset[c][1] = f
train_set = [
    (dataset[i][0], dataset[i][1]) for i in range(dataset_len) 
]
batch_size = 10 #prtotype
train_loader = DataLoader(
    train_set, batch_size= batch_size ,shuffle=True
)



pickle.dump(restore_AI,f'{directory}/AI/restore_AI.sav')