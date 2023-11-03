import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch.nn as nn
from mne.preprocessing import ICA
import mne
import matplotlib.pyplot as plt
from itertools import chain


FOLDER_PATH = './data_preprocessed_python'

def get_file_list(folder_path):
    dat_files = glob.glob(os.path.join(folder_path, '**', '*dat'), recursive=True)
    dat_files = sorted(dat_files)
    return dat_files


class ExtractedDeapDataset(Dataset):
    def __init__(self, dat_path, transform=None):
        self.dat_path = dat_path
        self.transform = transform
        self.data_length, self.dataX, self.dataY = self._pre_process()
    
    def _pre_process(self):
        
        with open(self.dat_path, 'rb') as f:
            data_pk = pickle.load(f, encoding='latin1')
        # print(data_pk.keys())
        
        self.dataX = data_pk['data']
        self.dataY = data_pk['labels']
        self.data_length = len(self.dataX)
        
        return self.data_length, self.dataX, self.dataY
    
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        data = self.dataX[idx]
        labels = self.dataY[idx]
        
        if self.transform:
            data = self.transform(data)

        return data, labels


class Simple1DCNN(nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 8, 64)  # Adjusted based on the input size
        self.fc2 = nn.Linear(64, 2)  # Output size 2 for binary classification

    def forward(self, x):
        x = x.view(-1, 1, 16)  # Reshape input to (batch_size, channels, length)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 8)  # Reshape based on the output size of the convolutional layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
        # 4, 1, 16
        # 4, 16, 16
        # 4, 16, 8
        # 4, 16 * 8
        # 4, 64
        # 4, 2
        
    
    
def training_epoch(model, data_loader, criterion, optimizer, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    total_loss = 0.0
    total_acc = 0.0
    total_sample = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for batch_idx, (data, target) in enumerate(data_loader):
        # import IPython; IPython.embed()
        # data = data.view(-1, 32, 32)
        # convert data, target to Tensor and move them to device (GPU), convert as a DoubleTensor
        target = target[:, 0] # get the first column of the target as a valence
        
        data, target = data.float().to(device), target.long().to(device)
        # convert to FloatTensor
        # data, target = data.float(), target.float()
        # import IPython; IPython.embed()
        # print(f"Batch idx: {batch_idx} data shape: {data.shape} target shape: {target.shape}, target: {target}")
        
        optimizer.zero_grad()
        # print(f"Batch idx: {batch_idx}")
        
        output = model(data)
        loss = criterion(output, target)
        
        if is_train:    
            loss.backward()
            optimizer.step()
        
        total_sample += len(data)
        total_loss += loss.item()
        _, pred = torch.max(output, 1)
        total_acc += (pred == target).sum().item()
    
        
    # train_loss = train_loss / len(data_loader.dataset)
    avg_loss = total_loss / total_sample
    avg_acc = total_acc / total_sample
    if is_train:
        print(f"Training total sample: {total_sample} Avg loss: {avg_loss} Avg acc: {avg_acc}")
    else:
        print(f"Test total sample: {total_sample} Avg loss: {avg_loss} Avg acc: {avg_acc}")
    # print(f"Total sample: {total_sample} Avg loss: {avg_loss} Avg acc: {avg_acc}")
    # return model, train_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    


if __name__ == "__main__":
    # import argparse
    # param = argparse.ArgumentParser()
    # param.add_argument("--seed", type=int, default=20)
    # args = param.parse_args()
    # set_seed(args.seed)
    for seed in range(199, 200):
        print(f"Seed: {seed}")
        set_seed(seed)
        
        dataset1 = ExtractedDeapDataset('./data_preprocessed_python/s01_processed.pkl')
        # print(f"Length of dataset1: {len(dataset1)}")
        
        # split dataset1 as a training set and a validation set
        train_size = int(0.8 * len(dataset1))
        val_size = len(dataset1) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset1, [train_size, val_size])
        # print(f"Length of train_dataset: {len(train_dataset)}")
        # print(f"Length of val_dataset: {len(val_dataset)}")
        print(train_dataset[0][0].shape)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
        # exit(0)
        
        simple_cnn = Simple1DCNN()
        # print(simple_cnn)
        simple_cnn.cuda()
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(simple_cnn.parameters(), lr=0.001)
        for epoch in range(10):
            print(f"Epoch: {epoch}")
            training_epoch(simple_cnn, train_loader, criterion, optimizer)
            training_epoch(simple_cnn, val_loader, criterion, optimizer, is_train=False)
    