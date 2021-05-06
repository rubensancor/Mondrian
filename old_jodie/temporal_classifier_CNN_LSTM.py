from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import argparse
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', required=True, help='Data path')
parser.add_argument('-ws', '--wandb_sync', '--wandb_sync=1', action='store_true', help='Check if the run is going to be uploaded to WandB')
args = parser.parse_args()

if not args.wandb_sync:
    os.environ['WANDB_MODE'] = 'dryrun'

wandb.init(project="mondrian", config=args, tags='Line-classificator')


random_seed = 42
np.random.seed(random_seed)

print('*** Loading dataset ***')
dataset_name = args.dataset
feature_tensor = torch.load(dataset_name + '_features.pt')
label_tensor = torch.load(dataset_name + '_labels.pt')
print('*** Dataset loaded! ***')

dataset = TensorDataset(
    feature_tensor,
    label_tensor
)

batch_size = 16
shuffle_dataset = True
validation_split = .2

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=val_sampler)
                                        
class TimeClassifier(nn.Module):
    def __init__(self, timelines, dense_size_1, dense_size_2, kernel_heights, selection='LSTM' ,out_channels=64, dropout_prob=.5):
        super(TimeClassifier, self).__init__()

        #time steps
        #time size
        self.feat_size = int(timelines.size(2))
        self.dropout_prob = dropout_prob
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.dense_size_1 = dense_size_1
        self.dense_size_2 = dense_size_2
        self.selection = selection

        if selection == 'CNN':
            __CNN__(self)
        elif selection == 'LSTM':
            __LSTM__(self)

        self.dense_1 = nn.Linear(len(self.kernel_heights) * out_channels, self.dense_size_1)
        self.dropout_1 = nn.Dropout(self.dropout_prob)

        self.dense_2 = nn.Linear(self.dense_size_1, self.dense_size_2)
        self.dropout_2 = nn.Dropout(self.dropout_prob)
        
        self.dense_soft = nn.Linear(self.dense_size_2, 2)

    def __CNN__(self):

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                out_channels=self.out_channels,
                                kernel_size=(kernel_h, self.feat_size)) 
                                for kernel_h in self.kernel_heights])


    def __LSTM__(self):





    def forward(self, x):
        x = x.unsqueeze(1)
        
        convs = [self.conv_block(x, conv) for conv in self.convs]
        x = torch.cat(convs, dim=1)
                
        x = self.dense_1(x)
        x = F.relu(x)
        x = self.dropout_1(x)
        
        x = self.dense_2(x)
        x = F.relu(x)
        x = self.dropout_2(x)

        x = self.dense_soft(x)
        
        return x

        
    def conv_block(self, input, conv_layer):

        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim)
        return F.max_pool1d(activation, activation.size(2)).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
epochs = 3000
num_users = feature_tensor.size(0)
loss_history = []
acc_history = []

model = TimeClassifier(feature_tensor, 128, 64, kernel_heights=[4,4,4]).to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=lr)

wandb.watch(model)

print('*** Starting the experiment ***')
for epoch in range(1,epochs + 1):
    model.train()
    epoch_loss = 0

    for batch_idx, (timeline, label) in enumerate(train_loader):
        timeline = timeline.to(device)
        label = label.to(device)
        
        opt.zero_grad()
        out = model(timeline)
    
        loss = criterion(out, label)
        epoch_loss += loss.item()
        loss.backward()
        opt.step
    
    epoch_loss /= num_users 
    loss_history.append(epoch_loss)

    model.eval()
    correct, total = 0, 0
    for batch_idx, (timeline, label) in enumerate(validation_loader):
        timeline = timeline.to(device)
        label = label.to(device)

        out = model(timeline)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += label.size(0)
        correct += (preds == label).sum().item()
        
    acc = correct / total
    acc_history.append(acc)
    print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
    wandb.log({'epoch': epoch,
               'loss': epoch_loss,
               'acc': acc})