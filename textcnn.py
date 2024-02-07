#%%
import math

import torch
import torch.nn as nn
import torch

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import torch.nn.functional as F


class TextCNN(pl.LightningModule):
    """Text classification model by character CNN, the implementation of paper
    'Yoon Kim. 2014. Convolution Neural Networks for Sentence
    Classification.'
    """

    def __init__(self, vocab_size, padding_idx,  \
                kernel_nums, kernel_sizes, embed_dim, hidden_dim, drop_prob):
        super(TextCNN, self).__init__()
        self.save_hyperparameters()

        # vocab_size = vocab_size
        # padding_idx = padding_idx
        # kernel_nums = [100, 100, 100]
        # kernel_sizes = [3, 4, 5]
        # embed_dim = 300
        # hidden_dim = 100
        # drop_prob = 0.5
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # no support for pre-trained embedding currently
        self.embed.padding_idx = padding_idx
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, kn, ks)
             for kn, ks in zip(kernel_nums, kernel_sizes)])
        self.fc = nn.Linear(sum(kernel_nums), hidden_dim)
        self.drop = nn.Dropout(drop_prob)
        self.out = nn.Linear(hidden_dim, 1)

        self.loss = nn.MSELoss()

    def forward(self, word_seq):
        # embed
        # print(word_seq)
        e = self.drop(self.embed(word_seq.to(torch.long)))  # [b,msl]->[b,msl,e]

        # conv and pool, [b,msl,e]->[b,h,msl]
        e = e.transpose(1, 2)  # [b,msl,e]->[b,e,msl]
        ps = []
        for conv in self.convs:
            c = conv(e)  # [b,e,msl]->[b,h,msl-k]
            p = F.max_pool1d(c, kernel_size=c.size(-1)).squeeze(-1)  # [b,h]
            ps.append(p)
        p = torch.cat(ps, dim=1)  # [b,h]

        # feed-forward, [b,h]->[b]
        f = self.drop(self.fc(p))
        logits = self.out(f).squeeze(-1)
        return logits

    def training_step(self, batch, batch_idx):
        src, tgt = batch['ids'], batch['hl']
        # print(tgt.dtype)
        output = self(src)
        # print(output.dtype, output.shape)

        loss = self.loss(tgt, output)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt = batch['ids'], batch['hl']
        output = self(src)
        loss = self.loss(tgt, output)
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

