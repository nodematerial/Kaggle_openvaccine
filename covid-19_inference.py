#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import warnings
warnings.filterwarnings('ignore')
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


# In[2]:


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


# In[3]:


set_seed(2020)


# In[4]:


test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
samplesub= pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


# In[5]:


bpp_max=[]
bpp_mean =[]

id = test.id.values
for i in id:
    probability = np.load('../input/stanford-covid-vaccine'+'/bpps/%s.npy'%i)
    bpp_max.append(probability.max(-1).tolist())
    bpp_mean.append(probability.mean(-1).tolist())
test['bpp_max']=bpp_max
test['bpp_mean']=bpp_mean


# In[6]:


test_public=test[test['seq_length']==107]
test_private=test[test['seq_length']==130]


# In[7]:


test_public_x=test_public.loc[:,['id','sequence','structure','predicted_loop_type','bpp_max','bpp_mean']]
test_private_x=test_private.loc[:,['id','sequence','structure','predicted_loop_type','bpp_max','bpp_mean']]
#CUDAに乗らないので、privateデータのサイズを小さくする。
test_private_x1,test_private_x2=train_test_split(test_private_x,test_size=0.5)


# In[8]:


token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
def preprocess_inputs_public(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea= np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    bpps_max_fea = np.array(test_public_x['bpp_max'].to_list())[:,:,np.newaxis]
    bpps_mean_fea = np.array(test_public_x['bpp_mean'].to_list())[:,:,np.newaxis]
    return np.concatenate([base_fea,bpps_max_fea,bpps_mean_fea], 2)

def preprocess_inputs_private1(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea= np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    bpps_max_fea = np.array(test_private_x1['bpp_max'].to_list())[:,:,np.newaxis]
    bpps_mean_fea = np.array(test_private_x1['bpp_mean'].to_list())[:,:,np.newaxis]
    return np.concatenate([base_fea,bpps_max_fea,bpps_mean_fea], 2)

def preprocess_inputs_private2(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    base_fea= np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )
    bpps_max_fea = np.array(test_private_x2['bpp_max'].to_list())[:,:,np.newaxis]
    bpps_mean_fea = np.array(test_private_x2['bpp_mean'].to_list())[:,:,np.newaxis]
    return np.concatenate([base_fea,bpps_max_fea,bpps_mean_fea], 2)


# In[9]:


test_public_inputs = torch.from_numpy(preprocess_inputs_public(test_public_x)).to(device).float()
test_private_inputs1 = torch.from_numpy(preprocess_inputs_private1(test_private_x1)).to(device).float()
test_private_inputs2 = torch.from_numpy(preprocess_inputs_private2(test_private_x2)).to(device).float()


# In[10]:


#print('train_入力：{}\nvalue_入力：{}\ntrain_ラベル：{}\nvalue_ラベル：{}'.format(train_inputs.shape,val_inputs.shape,train_outputs.shape,val_outputs.shape))


# In[11]:


class LSTM_model(nn.Module):
    def __init__(
        self, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=1024, hidden_layers=2
    ):
        super(LSTM_model, self).__init__()
        self.pred_len = pred_len

        self.embeding = nn.Embedding(num_embeddings=len(token2int), embedding_dim=embed_dim)
        self.lstm = nn.LSTM(
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
        output, hidden = self.lstm(reshaped)
        truncated = output[:, : self.pred_len, :]
        out = self.linear(truncated)
        return out


# In[12]:


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


# In[13]:


LSTM_weights_path='../input/weight11/LSTM_ver20.pth'

def get_LSTM_model(seq_len=107, pred_len=68):
    model = LSTM_model(seq_len=seq_len, pred_len=pred_len)
    checkpoint = torch.load(LSTM_weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    return model


# In[14]:


GRU_weights_path='../input/weight11/GRU_ver8'

def get_GRU_model(seq_len=107, pred_len=68):
    model = GRU_model(seq_len=seq_len, pred_len=pred_len)
    checkpoint = torch.load(GRU_weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    return model


# In[15]:


with torch.no_grad():
    model =get_LSTM_model()
    prediction=model(test_public_inputs)
    result_public_LSTM=prediction.to('cpu').detach().numpy().copy()
del prediction

with torch.no_grad():
    model =get_LSTM_model(seq_len=130, pred_len=91)
    prediction=model(test_private_inputs1)
    result_private1_LSTM=prediction.to('cpu').detach().numpy().copy()
del prediction

with torch.no_grad():
    model =get_LSTM_model(seq_len=130, pred_len=91)
    prediction=model(test_private_inputs2)
    result_private2_LSTM=prediction.to('cpu').detach().numpy().copy()
del prediction


# In[16]:


with torch.no_grad():
    model =get_GRU_model()
    prediction=model(test_public_inputs)
    result_public_GRU=prediction.to('cpu').detach().numpy().copy()
del prediction

with torch.no_grad():
    model =get_GRU_model(seq_len=130, pred_len=91)
    prediction=model(test_private_inputs1)
    result_private1_GRU=prediction.to('cpu').detach().numpy().copy()
del prediction

with torch.no_grad():
    model =get_GRU_model(seq_len=130, pred_len=91)
    prediction=model(test_private_inputs2)
    result_private2_GRU=prediction.to('cpu').detach().numpy().copy()
del prediction


# In[17]:


df0 = pd.DataFrame(index=range(39), columns=['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',])
df0=df0.fillna(0)


# In[18]:


test_public_id=test_public['id']
idlist_public=test_public_id.values.tolist()


# In[19]:


test_private_id1=test_private_x1['id']
idlist_private1=test_private_id1.values.tolist()
idlist_private1[-5:]


# In[20]:


test_private_id2=test_private_x2['id']
idlist_private2=test_private_id2.values.tolist()
idlist_private2[:5]


# In[21]:


#無理やりソートすることに
testindex=samplesub.loc[:,['id_seqpos']]
testindex=testindex.reset_index()


# In[22]:


df1 = pd.DataFrame(result_public_LSTM[0])
df1.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
df1.insert(0, 'id_seqpos', 0)
df1=pd.concat([df1,df0])
id=idlist_public[0]
for i in range(len(df1)):
    df1.iloc[i,0]=id+'_{}'.format(i)
for j in range (len(result_public_LSTM)-1):
    id = idlist_public[j+1]
    df2 = pd.DataFrame(result_public_LSTM[j+1])
    df2.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
    df2.insert(0, 'id_seqpos', 0)
    df2=pd.concat([df2,df0]) 
    for i in range(len(df2)):
        df2.iloc[i,0]=id+'_{}'.format(i)
    df1=pd.concat([df1,df2])
public_dataframe=df1

df1 = pd.DataFrame(result_private1_LSTM[0])
df1.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
df1.insert(0, 'id_seqpos', 0)
df1=pd.concat([df1,df0])
id=idlist_private1[0]
for i in range(len(df1)):
    df1.iloc[i,0]=id+'_{}'.format(i)
for j in range (len(result_private1_LSTM)-1):
    id = idlist_private1[j+1]
    df2 = pd.DataFrame(result_private1_LSTM[j+1])
    df2.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
    df2.insert(0, 'id_seqpos', 0)
    df2=pd.concat([df2,df0])
    for i in range(len(df2)):
        df2.iloc[i,0]=id+'_{}'.format(i)
    df1=pd.concat([df1,df2])
private_dataframe1=df1

df1 = pd.DataFrame(result_private2_LSTM[0])
df1.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
df1.insert(0, 'id_seqpos', 0)
df1=pd.concat([df1,df0])
id=idlist_private2[0]
for i in range(len(df1)):
    df1.iloc[i,0]=id+'_{}'.format(i)
for j in range (len(result_private2_LSTM)-1):
    id = idlist_private2[j+1]
    df2 = pd.DataFrame(result_private2_LSTM[j+1])
    df2.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
    df2.insert(0, 'id_seqpos', 0)
    df2=pd.concat([df2,df0])
    for i in range(len(df2)):
        df2.iloc[i,0]=id+'_{}'.format(i)
    df1=pd.concat([df1,df2])
private_dataframe2=df1


# In[23]:


merged_dataframe=pd.concat([public_dataframe,private_dataframe1,private_dataframe2])

pre_submission_LSTM=pd.merge(testindex,merged_dataframe)


# In[24]:


pre_submission_LSTM


# In[25]:


df1 = pd.DataFrame(result_public_GRU[0])
df1.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
df1.insert(0, 'id_seqpos', 0)
df1=pd.concat([df1,df0])
id=idlist_public[0]
for i in range(len(df1)):
    df1.iloc[i,0]=id+'_{}'.format(i)
for j in range (len(result_public_GRU)-1):
    id = idlist_public[j+1]
    df2 = pd.DataFrame(result_public_GRU[j+1])
    df2.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
    df2.insert(0, 'id_seqpos', 0)
    df2=pd.concat([df2,df0]) 
    for i in range(len(df2)):
        df2.iloc[i,0]=id+'_{}'.format(i)
    df1=pd.concat([df1,df2])
public_dataframe=df1

df1 = pd.DataFrame(result_private1_GRU[0])
df1.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
df1.insert(0, 'id_seqpos', 0)
df1=pd.concat([df1,df0])
id=idlist_private1[0]
for i in range(len(df1)):
    df1.iloc[i,0]=id+'_{}'.format(i)
for j in range (len(result_private1_GRU)-1):
    id = idlist_private1[j+1]
    df2 = pd.DataFrame(result_private1_GRU[j+1])
    df2.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
    df2.insert(0, 'id_seqpos', 0)
    df2=pd.concat([df2,df0])
    for i in range(len(df2)):
        df2.iloc[i,0]=id+'_{}'.format(i)
    df1=pd.concat([df1,df2])
private_dataframe1=df1

df1 = pd.DataFrame(result_private2_GRU[0])
df1.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
df1.insert(0, 'id_seqpos', 0)
df1=pd.concat([df1,df0])
id=idlist_private2[0]
for i in range(len(df1)):
    df1.iloc[i,0]=id+'_{}'.format(i)
for j in range (len(result_private2_GRU)-1):
    id = idlist_private2[j+1]
    df2 = pd.DataFrame(result_private2_GRU[j+1])
    df2.columns = ['reactivity', 'deg_Mg_pH10', 'deg_pH10','deg_Mg_50C','deg_50C',]
    df2.insert(0, 'id_seqpos', 0)
    df2=pd.concat([df2,df0])
    for i in range(len(df2)):
        df2.iloc[i,0]=id+'_{}'.format(i)
    df1=pd.concat([df1,df2])
private_dataframe2=df1


# In[26]:


merged_dataframe=pd.concat([public_dataframe,private_dataframe1,private_dataframe2])

pre_submission_GRU=pd.merge(testindex,merged_dataframe)


# In[27]:


blend_preds_df = pd.DataFrame()
blend_preds_df['id_seqpos']=pre_submission_GRU['id_seqpos']
blend_preds_df['reactivity'] = .5*pre_submission_GRU['reactivity'] + .5*pre_submission_LSTM['reactivity']
blend_preds_df['deg_Mg_pH10'] = .5*pre_submission_GRU['deg_Mg_pH10'] + .5*pre_submission_LSTM['deg_Mg_pH10']
blend_preds_df['deg_pH10'] = .5*pre_submission_GRU['deg_pH10'] + .5*pre_submission_LSTM['deg_pH10']
blend_preds_df['deg_Mg_50C'] = .5*pre_submission_GRU['deg_Mg_50C'] + .5*pre_submission_LSTM['deg_Mg_50C']
blend_preds_df['deg_50C'] = .5*pre_submission_GRU['deg_50C'] + .5*pre_submission_LSTM['deg_50C']
blend_preds_df


# In[28]:


blend_preds_df.to_csv("submission.csv", index=False)

