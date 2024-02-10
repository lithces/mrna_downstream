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


# Positional Encoding module
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(pl.LightningModule):
    def __init__(self, vocab_sz, output_dim, hidden_dim, num_layers, num_heads, dropout_rate, itm_dim, comments=""):
        super(TransformerModel, self).__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_sz, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout_rate)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, itm_dim, dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.comments = comments
        # self.linear = nn.Linear(hidden_dim, output_dim)
        self.mlp = nn.Linear(hidden_dim, output_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, hidden_dim // 4),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 4, hidden_dim // 8),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 8, output_dim)
        # )
        # self.init_weights()
        self.loss = nn.MSELoss()
    def init_weights(self):
        initrange = 0.1    
        self.mlp.bias.data.zero_()
        self.mlp.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask):
        src = src.transpose(0,1).contiguous().to(torch.int)
        src = self.pos_encoder(self.embedding(src))
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.mlp(output)
        output = output.transpose(0,1)[:,:,0]
        keep_ind = (~src_key_padding_mask).to(torch.float)
        output = (output*keep_ind).sum(dim=-1) / keep_ind.sum(dim=-1)

        return output
    
    def training_step(self, batch, batch_idx):
        src, tgt, mask = batch['ids'], batch['hl'], batch['mask']
        # print(tgt.dtype)
        output = self(src, mask)
        # print(output.dtype, output.shape)

        loss = self.loss(tgt, output)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt, mask= batch['ids'], batch['hl'], batch['mask']
        output = self(src, mask)
        loss = self.loss(tgt, output)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

