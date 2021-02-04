#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import warnings
warnings.filterwarnings('ignore')
import wandb
import pandas as pd, numpy as np
import math, json, gc, random, os, sys
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from catalyst.dl import SupervisedRunner
from catalyst.contrib.dl.callbacks import WandbLogger
from contextlib import contextmanager
from catalyst.dl.callbacks import AccuracyCallback, F1ScoreCallback, OptimizerCallback
#from pytorch_memlab import profile, MemReporter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[3]:


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


# In[4]:


set_seed(2020)


# In[5]:


train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
train=train[train['SN_filter']==1]


# In[6]:


bpp_max =[]
bpp_sum =[]

id = train.id.values
for i in id:
    probability = np.load('../input/stanford-covid-vaccine'+'/bpps/%s.npy'%i)
    bpp_max.append(probability.max(-1).tolist())
    bpp_sum.append(probability.sum(-1).tolist())


# In[7]:


train['bpp_max']=bpp_max
train['bpp_sum']=bpp_sum


# In[8]:


trainval_x=train.loc[:,['id','sequence','structure','predicted_loop_type','bpp_max','bpp_sum']]
trainval_y=train.loc[:,['reactivity','deg_Mg_pH10','deg_pH10','deg_Mg_50C','deg_50C']]
trainval_x=trainval_x.reset_index(drop=True)
trainval_y=trainval_y.reset_index(drop=True)


# In[9]:


train_x,val_x,train_y,val_y=train_test_split(trainval_x,trainval_y,test_size=0.2)


# In[10]:


token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
def preprocess_inputs_train(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea= np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    bpps_max_fea = np.array(train_x['bpp_max'].to_list())[:,:,np.newaxis]
    bpps_sum_fea = np.array(train_x['bpp_sum'].to_list())[:,:,np.newaxis]
    return np.concatenate([base_fea,bpps_max_fea,bpps_sum_fea], 2)

def preprocess_inputs_val(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea= np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    bpps_max_fea = np.array(val_x['bpp_max'].to_list())[:,:,np.newaxis]
    bpps_sum_fea = np.array(val_x['bpp_sum'].to_list())[:,:,np.newaxis]
    return np.concatenate([base_fea,bpps_max_fea,bpps_sum_fea], 2)


# In[11]:


train_y_array= np.array(train_y.values.tolist()).transpose(0, 2, 1)
val_y_array= np.array(val_y.values.tolist()).transpose(0, 2, 1)


# In[12]:


train_inputs = torch.from_numpy(preprocess_inputs_train(train_x))
val_inputs = torch.from_numpy(preprocess_inputs_val(val_x))
train_outputs=torch.tensor(train_y_array).clone().double()
val_outputs=torch.tensor(val_y_array).clone().double()


# In[13]:


class DataSet:
    def __init__(self,X,Y):
        self.X = X# 入力
        self.t = Y# 出力

    def __len__(self):
        return len(self.X) 

    def __getitem__(self, index):
        return self.X[index], self.t[index]


# In[14]:


class GRU_model(nn.Module):
    def __init__(
        self, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=1024, hidden_layers=2
    ):
        super(GRU_model, self).__init__()
        self.pred_len = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim * 3+2,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_dim * 2, 5)

    def forward(self, seqs):
        embed = self.embeding(seqs[:,:,0:3].long())
        reshaped = torch.reshape(embed, (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))
        reshaped= torch.cat((reshaped,seqs[:,:,3:5]),2)
        output, hidden = self.gru(reshaped)
        truncated = output[:, : self.pred_len, :]
        out = self.linear(truncated)
        return out


# In[15]:


loaders = {
    "train": data.DataLoader(DataSet(train_inputs,train_outputs), 
                             batch_size=32, 
                             shuffle=True, 
                             num_workers=2, 
                             pin_memory=True, 
                             drop_last=True),
    "valid": data.DataLoader(DataSet(val_inputs,val_outputs), 
                             batch_size=32, 
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True,
                             drop_last=False)
}

output_path = './'

# model
model = GRU_model().to(device).double()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=33)

# Loss
criterion = nn.MSELoss()

runner = SupervisedRunner(
    device=device,)

runner.train(
    model=model,
    criterion=nn.MSELoss(),
    loaders=loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=30,
    verbose=True,
    logdir=output_path,
    callbacks=[WandbLogger(project="GRU-project",name= 'train-7(h=512,l=2)')],
)

