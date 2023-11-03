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


class DeapDataset(Dataset):
    def __init__(self, dat_path, transform=None):
        self.dat_path = dat_path
        self.transform = transform
        self.data_length, self.dataX, self.dataY = self._pre_process()
        
    
    def SignalPreProcess(self, eeg_rawdata):
        """
        :param eeg_rawdata: numpy array with the shape of (n_channels, n_samples)
        :return: filtered EEG raw data
        """
        N_C = None
        droping_components = 'one'
        
        assert eeg_rawdata.shape[0] == 32
        eeg_rawdata = np.array(eeg_rawdata)

        ch_names = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", 
                    "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
                    "PO4", "O2"]
    
        info = mne.create_info(ch_names = ch_names, ch_types = ['eeg' for _ in range(32)], sfreq = 128, verbose=False)
        raw_data = mne.io.RawArray(eeg_rawdata, info, verbose = False)
        raw_data.load_data(verbose = False).filter(l_freq = 4, h_freq = 48, method = 'fir', verbose = False)
        #raw_data.plot()

        ica = ICA(n_components = N_C, random_state = 97, verbose = False)
        ica.fit(raw_data)
        # https://mne.tools/stable/generated/mne.preprocessing.find_eog_events.html?highlight=find_eog_#mne.preprocessing.find_eog_events
        eog_indices, eog_scores = ica.find_bads_eog(raw_data.copy(), ch_name = 'Fp1', verbose = None)
        a = abs(eog_scores).tolist()
        if(droping_components == 'one'):
            ica.exclude = [a.index(max(a))]
            
        else: # find two maximum scores
            a_2 = a.copy()
            a.sort(reverse = True)
            exclude_index = []
            for i in range(0, 2):
                for j in range(0, len(a_2)):
                    if(a[i]==a_2[j]):
                        exclude_index.append(j)
            ica.exclude = exclude_index
        ica.apply(raw_data, verbose = False)
        # common average reference
        raw_data.set_eeg_reference('average', ch_type = 'eeg')#, projection = True)
        filted_eeg_rawdata = np.array(raw_data.get_data())
        return filted_eeg_rawdata

    
    def signal_pro(self, input_data):
        print(f"Performing signal processing... {input_data.shape}")
        for i in range(input_data.shape[0]):
            input_data[i] = self.SignalPreProcess(input_data[i].copy())
        print(f"Data after signal processing: {input_data.shape}")
        return input_data 
    
    def get_feature(self, data):
        channel_no = [0, 2, 16, 19] # only taking these four channels
        feature_vector = [6.2, 7.3, 6.2, 7.3]
        counter_patient = 0
        total_channel_no = 4
        feature = np.ones(40*len(feature_vector)*total_channel_no).reshape(40,len(feature_vector)*total_channel_no)
        feature_matrix = []
        for ith_video in range(40):
            features = []
            for ith_channel in channel_no:
                # power spectral density
                # please refer: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.psd.html
                psd, freqs = plt.psd(data[ith_video][ith_channel], Fs = 128)
                # get frequency bands mean power
                theta_mean = np.mean(psd[np.logical_and(freqs >= 4, freqs <= 7)])
                alpha_mean = np.mean(psd[np.logical_and(freqs >= 8, freqs <= 13)])
                beta_mean  = np.mean(psd[np.logical_and(freqs >= 13, freqs <= 30)])
                gamma_mean = np.mean(psd[np.logical_and(freqs >= 30, freqs <= 40)])
                features.append([theta_mean, alpha_mean, beta_mean, gamma_mean])
            # flatten the features i.e. transform it from 2D to 1D
            feature_matrix.append(np.array(list(chain.from_iterable(features))))
        return np.array(feature_matrix)
    
    def _pre_process(self):
        with open(self.dat_path, 'rb') as f:
            dat_loader = pickle.load(f, encoding='latin1')
        
        # import IPython; IPython.embed()
        data = dat_loader['data']
        labels = dat_loader['labels']
        n_video, n_channel, n_time = data.shape
        
        # get data from 32 channels
        data = data[:, 0:32, 384:8064]
        data = self.signal_pro(data)
        data = self.get_feature(data)
        # write numpy array to pickle file
        
        # new_fn = self.dat_path.split('/')[-1].split('.')[0] + '_processed.pkl'
        # new_fn = os.path.join('./data_preprocessed_python', new_fn)
        new_fn = self.dat_path.replace('.dat', '_processed.pkl')
        
            
        # import IPython; IPython.embed()
        # exit(0)
        
        valence, arousal, _, _ = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]
        # 40 videos, 32 channels, 8064 time
        
        
        # data = data.astype('float').reshape(-1, 40)
        self.data_length = n_video
        self.dataX = data
        
        # set threshold for valence and arousal
        # valence[valence < 5] = 0
        # valence[valence >= 5] = 1
        # arousal[arousal < 5] = 0
        # arousal[arousal >= 5] = 1
        
        valence_labels = np.array([1 if v >= 5 else 0 for v in valence])
        arousal_labels = np.array([1 if a >= 5 else 0 for a in arousal])
        four_labels = np.array([0 if v >= 5 and a >= 5 else 1 if v >= 5 and a < 5 else 2 if v < 5 and a >= 5 else 3 for v, a in zip(valence, arousal)])
        
        self.dataY = np.concatenate((valence_labels.reshape(-1, 1), arousal_labels.reshape(-1, 1), np.array(four_labels).reshape(-1, 1)), axis=1)
        # write dataX, dataY to pickle file with key: data, labels
        with open(new_fn, 'wb') as f:
            pickle.dump({'data': self.dataX, 'labels': self.dataY}, f)
        
        
        
        # valence (40) -> (40, 1) [ [1], [2] ] -> (40, 2) [ [1, 3], [ ],  ]
        return self.data_length, self.dataX, self.dataY
    
    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        data = self.dataX[idx]
        labels = self.dataY[idx]
        
        if self.transform:
            data = self.transform(data)

        return data, labels

# Define a simple CNN model
class Simple2DCNN(nn.Module):
    def __init__(self):
        super(Simple2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 6 * 2014, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 32, 8064)
        # import IPython; IPython.embed()
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 2014)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
        # 4, 1, 32, 8064
        # 4, 16, 30, 8062
        # 4, 16, 15, 4031
        # 4, 32, 13, 4029
        # 4, 32, 6, 2014
        # 4, 32 * 6 * 2014
        # 4, 128
        # 4, 10

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
    
    set_seed(42)
    
    # dataset1 = DeapDataset('./data_preprocessed_python/s01.dat')
    for i in range(2, 33):
        dataset1 = DeapDataset(f'./data_preprocessed_python/s{i:02d}.dat')
        print(f"Length of dataset: {len(dataset1)}")
        
    exit(0)
    
    # split dataset1 as a training set and a validation set
    train_size = int(0.8 * len(dataset1))
    val_size = len(dataset1) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset1, [train_size, val_size])
    print(f"Length of train_dataset: {len(train_dataset)}")
    print(f"Length of val_dataset: {len(val_dataset)}")
    print(train_dataset[0][0].shape)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
    
    simple_cnn = SimpleCNN()
    print(simple_cnn)
    simple_cnn.cuda()
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(simple_cnn.parameters(), lr=0.001)
    for epoch in range(10):
        print(f"Epoch: {epoch}")
        training_epoch(simple_cnn, train_loader, criterion, optimizer)
        training_epoch(simple_cnn, val_loader, criterion, optimizer, is_train=False)
        # print(f"Epoch: {epoch} Training Loss: {train_loss}")
        
    
    
    # list_of_files = get_file_list(FOLDER_PATH)
    # for idx, file in enumerate(list_of_files, 1):
    #     print(f"Process file with: {idx:02d} and name: {file}")

    #     dataset = DeapDataset(file)
    #     print(f"Length of dataset: {len(dataset)}")
    #     # for id in range(10):
        #     print(f"id: {id}")
        #     data_idx, labels_idx = dataset[id]
        #     print(f"Shape of data: {data_idx.shape} label: {labels_idx}")
        
    