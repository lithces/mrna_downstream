#%%
vocab_sz = 140
padding_idx = 139
output_dim = 1
hidden_dim = 64
num_layers = 3
batch_size = 256
max_epochs = 200
dropout = None
ctx_size = 4096 
ignore_input_ids = False
comments = f"ctx_size: {ctx_size}" # default 4096

lr = 1e-4 # default 1e-3
opt = 'AdamW' # default Adam

from torch.utils.data import DataLoader, StackDataset
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


class EmbedSeqDataset(LocalDataset):
    def __init__(self, local, ctx_size):
        '''
        resulting tensors are padded to ctx_size
        '''
        super().__init__(local=local)
        self.ctx_size = ctx_size


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        embs = obj['embs']
        L,D = embs.shape
        pos1 = min(L, self.ctx_size)
        padded_emb = np.concatenate( (embs[:pos1], np.zeros( ( (self.ctx_size - pos1),D), dtype=np.float16)))
        
        return {'embs': padded_emb
        }


#%%
import tqdm
ds_train_base = BatchSeqDataset('./mds/tr', ctx_size=ctx_size)
ds_train_emb = EmbedSeqDataset('./mds_emb_mamba/tr', ctx_size=ctx_size)

ds_val_base = BatchSeqDataset('./mds/te', ctx_size=ctx_size)
ds_val_emb = EmbedSeqDataset('./mds_emb_mamba/te', ctx_size=ctx_size)

ds_train = StackDataset(ds_train_base, ds_train_emb)
ds_val = StackDataset(ds_val_base, ds_val_emb)

dl_debug = DataLoader(ds_train, shuffle=True, batch_size=256)

for di in tqdm.tqdm(dl_debug):
    print(di[1]['embs'].dtype)
    print(di[1]['embs'].shape)
    input_emb_dim = di[1]['embs'].shape[-1]
    break
    # pass
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# import pytorch_lightning as pl
import lightning.pytorch as pl
from mymamba_with_emb_input import *
#%%





dl_train = DataLoader(ds_train, shuffle=True, batch_size=batch_size)
dl_val = DataLoader(ds_val, shuffle=False, batch_size=batch_size)

from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.loggers import WandbLogger
# wandb_logger = WandbLogger(project="mamba_mrna_downstream_with_embs")

# logger = TensorBoardLogger(save_dir='training_log_mamba')
from lightning.pytorch.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(monitor="val_loss", \
    save_top_k=3, \
    mode="min",)

model = MambaSingleOutputModelWithEmbeddingInput(vocab_sz, output_dim, hidden_dim, num_layers, input_emb_dim, ignore_input_ids=ignore_input_ids\
                                                 , dropout_rate=dropout, comments=comments, lr=lr, opt=opt)
trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=5, val_check_interval=0.25, callbacks=[checkpoint_callback])
# trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=5, val_check_interval=0.25)

trainer.fit(model, dl_train, dl_val)
# %%
