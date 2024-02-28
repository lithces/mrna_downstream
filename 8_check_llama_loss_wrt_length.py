#%%
ctx_size = 64 
from torch.utils.data import DataLoader, StackDataset
from streaming import LocalDataset
import numpy as np

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
        
        return {'embs': padded_emb, 'L': obj['L'], 'loss': obj['loss']
        }


#%%
import tqdm
ds_val_emb = EmbedSeqDataset('./mds_emb_llama_rc_2/va', ctx_size=ctx_size)

#%%
dl_debug = DataLoader(ds_val_emb, shuffle=True, batch_size=1)

lst = []
for di in tqdm.tqdm(dl_debug):
    # pass
    lst.append([di['L'].item(), di['loss'].item()])


#%%
import pandas as pd
df = pd.DataFrame(data = lst, columns = ['L', 'loss'])
# %%
import seaborn as sns
sns.regplot(df, x='L', y='loss', lowess=True, line_kws={'color': 'red'})
# %%
