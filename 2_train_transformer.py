#%%
vocab_sz = 140
padding_idx = 139
output_dim = 1
hidden_dim = 64
num_layers = 3
num_heads = 8
dropout_rate = 0.1
itm_dim = 64
batch_size = 256
max_epochs = 200


import lightning as L
from torch.utils.data import DataLoader
from streaming import LocalDataset
import numpy as np

class OneSeqDataset(LocalDataset):
    def __init__(self, local):
        super().__init__(local=local)


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        return {'L': obj['L'] \
            ,'c_res_id' :obj['c_res_id'] \
            ,'c_ss_id' : obj['c_ss_id'] \
            ,'c_coord' : obj['c_coord'] \
            ,'c_chi' : obj['c_chi'] \
            ,'id_orig': obj['id_orig'] \
            ,'y': obj['y']
        }

class BatchSeqDataset(LocalDataset):
    def __init__(self, local, ctx_size=4096):
        '''
        resulting tensors are padded to ctx_size
        '''
        super().__init__(local=local)
        self.ctx_size = ctx_size


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        res_id0, y = obj['ids'], obj['targets'].astype(np.float32)[0] # only take halflife
        L = len(res_id0)
        # print(L, y)
        pos1 = min(L, self.ctx_size)
        res_id = np.concatenate( (res_id0[:pos1], np.array([padding_idx]*(self.ctx_size - pos1), dtype=np.uint8)))
        
        mask = np.concatenate( (np.array([0]*pos1, dtype=np.bool8), np.array([1]*(self.ctx_size - pos1),  dtype=np.bool8)))
        return {'L': L \
            ,'ids' :res_id \
            ,'mask': mask
            ,'hl': y
        }

#%%
import tqdm
ds_train = BatchSeqDataset('./mds/tr', ctx_size=4096)
ds_val = BatchSeqDataset('./mds/te', ctx_size=4096)

dl_debug = DataLoader(ds_train, shuffle=True, batch_size=256)

# for di in tqdm.tqdm(dl_debug):
#     # print(di['ids'].dtype)
#     # break
#     pass
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from mytransformer import *
#%%





dl_train = DataLoader(ds_train, shuffle=True, batch_size=batch_size)
dl_val = DataLoader(ds_val, shuffle=False, batch_size=batch_size)


model = TransformerModel(vocab_sz, output_dim, hidden_dim, num_layers, num_heads, dropout_rate, itm_dim)
trainer = pl.Trainer(max_epochs=max_epochs)
trainer.fit(model, dl_train, dl_val)
# %%
