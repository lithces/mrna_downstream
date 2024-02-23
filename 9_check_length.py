#%%
suffix = 'mds_unigram_1024'
ds_root = f'/home/lithtp/sanofi/mrna/mrna_downstream/{suffix}/'


#%%
from torch.utils.data import DataLoader
from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset
import numpy as np

class TokenTensorDataset(LocalDataset):
    def __init__(self, local):
        super().__init__(local=local)


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        dat = torch.tensor(obj['ids'].astype(np.int32))
        L = len(dat)
        return L

from mrna_utils import *
def create_dataloaders(batch_size, block_size):
    # Remote directory (S3 or local filesystem) where dataset is stored

    # Local directory where dataset is cached during operation
    lst_cate = [
        "tr"
        ,"va"
        ,"te"
    ]




    local_dirs = [ds_root+f'{catei}' for catei in lst_cate]
    lst_ds = [TokenTensorDataset(local=li) for li in local_dirs]
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

Ls = torch.concat(lst_ret)
#%%
import matplotlib.pyplot as plt
def plot_lens(Ls):
    a = Ls+0.0
    lst_pct = torch.arange(100)/100

    lst_th = torch.quantile(a, lst_pct)
    plt.scatter(lst_th, lst_pct)

# %%
plot_lens(Ls)
plt.title(f"{suffix}")
# plot_lens(L5s)
# plot_lens(L3s)
# plot_lens(L5s)
# %%
