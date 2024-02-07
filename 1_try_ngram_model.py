#%%
from torch.utils.data import DataLoader
from streaming import LocalDataset, StreamingDataset
import torch
from torch.utils.data import random_split, ConcatDataset
from mrna_utils import *

mymrnatok = MRNATOK()

#%%
class TokenTensorDataset(LocalDataset):
    def __init__(self, local):
        super().__init__(local=local)


    def __getitem__(self, index: int):
        obj = super().__getitem__(index)
        dat = torch.tensor(obj['ids'])
        y = obj['targets'][0]
        L = len(dat)
        return ' '.join([str(x) for x in dat.tolist()]), y


ds_tr = TokenTensorDataset('mds/tr')
dl_tr = DataLoader(ds_tr, batch_size=len(ds_tr))

ds_va = TokenTensorDataset('mds/va')
dl_va = DataLoader(ds_va, batch_size=len(ds_va))

ds_te = TokenTensorDataset('mds/te')
dl_te = DataLoader(ds_va, batch_size=len(ds_te))

#%%
from sklearn.feature_extraction.text import CountVectorizer
for dati in dl_tr:
    corpus_tr, y_tr = dati
    break

for dati in dl_va:
    corpus_va, y_va = dati
    break
for dati in dl_te:
    corpus_te, y_te = dati
    break


# %%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern=r"\b[0-9]+\b", ngram_range=(1,1), binary=False)
X_tr = vectorizer.fit_transform(corpus_tr)
feat_name = vectorizer.get_feature_names_out()

#%%
X_va = vectorizer.transform(corpus_va)
X_te = vectorizer.transform(corpus_te)
# %%
from sklearn.linear_model import enet_path, lasso_path
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
X_tr1 = scaler.fit_transform(X_tr)
X_va1 = scaler.transform(X_va)
X_te1 = scaler.transform(X_te)

# %%
# %%
from sklearn.metrics import r2_score
from sklearn import linear_model
for alpha in [100,1000,10000,100000, 1000000]:
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(X_tr1, y_tr)
    y_va_pred = clf.predict(X_va1)
    print(alpha, r2_score(y_va, y_va_pred))
# %%
