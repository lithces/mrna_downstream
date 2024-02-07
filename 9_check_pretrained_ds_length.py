#%%
from torch.utils.data import DataLoader
from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset

class TokenTensorDataset(LocalDataset):
    def __init__(self, local, ctx_size):
        super().__init__(local=local)
        self.ctx_size = ctx_size+1 # need to add one for AR nature.


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        dat = torch.tensor(obj['ids'])
        L = len(dat)
        if L < self.ctx_size:            
            padding = torch.ones(self.ctx_size-L, dtype=torch.int16)*(-1)
            return torch.concat( (dat.to(torch.int16), padding))
        else:
            return dat[:self.ctx_size]

from mrna_utils import *
def create_dataloaders(batch_size, block_size):
    # Remote directory (S3 or local filesystem) where dataset is stored

    # Local directory where dataset is cached during operation
    lst_cate = [
        "rabbits"
        ,"monotremes"
        ,"insectivores"
        ,"odd-toed_ungulates"
        ,"more_placentals"
        ,"bats" 
        ,"even-toed_ungulates"
        ,"primates"
        ,"carnivores"
        ,"rodents"
    ]
    ds_root = '/data2/data/mrna_llm/raw_seqs_cds_pos/np_after_tok_mds/'




    local_dirs = [ds_root+f'{catei}' for catei in lst_cate]
    lst_ds = [DebugTensorDataset(local=li) for li in local_dirs]
    ds = ConcatDataset(lst_ds)
    dl = DataLoader(ds, batch_size=batch_size)
    # lst_rngs = [torch.Generator().manual_seed(42) for catei in lst_cate]
    # lst_splits = [random_split(ds, [0.95, 0.05], generator=rng) for ds, rng in zip(lst_ds, lst_rngs)]
    # lst_ds_train = [si[0] for si in lst_splits]
    # lst_ds_val = [si[1] for si in lst_splits]

    # ds_train = ConcatDataset(lst_ds_train)
    # ds_val = ConcatDataset(lst_ds_val)

    # dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    # dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True)
    return dl

dl = create_dataloaders(1024, 1024)
#%%
import tqdm
lst_ret = [xi for xi in tqdm.tqdm(dl)]
#%%
Ls = [xi[0] for xi in lst_ret]
L5s = [xi[1] for xi in lst_ret]
L3s = [xi[2] for xi in lst_ret]
Lcs = [xi[3] for xi in lst_ret]

import matplotlib.pyplot as plt
def plot_lens(Ls):
    a = torch.cat(Ls)+0.0
    lst_pct = torch.arange(100)/100

    lst_th = torch.quantile(a, lst_pct)
    plt.scatter(lst_th, lst_pct)

# %%
plot_lens(Ls)
# plot_lens(L5s)
# plot_lens(L3s)
# plot_lens(L5s)
# %%
