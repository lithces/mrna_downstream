#%%
vocab_sz = 140
padding_idx = 139
output_dim = 1
hidden_dim = 16
num_layers = 1
batch_size = 256
max_epochs = 200
dropout = None
ctx_size = 1024*8 
output_agg = 'avg'
comments = f"ctx_size: {ctx_size}, hidden_dim: {hidden_dim}, output_agg: {output_agg}, layers: {num_layers}, batchsize: {batch_size}" # default 4096

lr = 1e-4 # default 1e-3
opt = 'AdamW' # default Adam

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
    def __init__(self, local, ctx_size):
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
ds_train = BatchSeqDataset('./mds/tr', ctx_size=ctx_size)
ds_val = BatchSeqDataset('./mds/te', ctx_size=ctx_size)

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
import lightning.pytorch as pl
from mymamba import *
#%%





dl_train = DataLoader(ds_train, shuffle=True, batch_size=batch_size)
dl_val = DataLoader(ds_val, shuffle=False, batch_size=batch_size)
model = MambaSingleOutputModel(vocab_sz, output_dim, hidden_dim, num_layers, dropout_rate=dropout, output_agg=output_agg, comments=comments, lr=lr, opt=opt)

#%%
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
tf_logger = TensorBoardLogger(save_dir='training_logs_mamba')


# mlf_logger = MLFlowLogger(experiment_name="lightning_logs_mamba", tracking_uri="file:./ml-runs", run_name=comments)
mlf_logger = MLFlowLogger(experiment_name="lightning_logs_mamba", tracking_uri="http://127.0.0.1:5000", run_name=comments, log_model=True)


from lightning.pytorch.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(monitor="val_loss", \
    save_top_k=3, \
    mode="min",)
trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=5, val_check_interval=0.25, callbacks=[checkpoint_callback], logger=[mlf_logger, tf_logger])

trainer.fit(model, dl_train, dl_val)
# %%
