import math
from functools import partial
import json
import os

from collections import namedtuple


import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor

from dataclasses import dataclass, field

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import *
from mamba_ssm.models.mixer_seq_simple import _init_weights

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class MambaSingleOutputModel(pl.LightningModule, GenerationMixin):

    def __init__(
        self,
        vocab_sz, output_dim, hidden_dim, num_layers, dropout_rate = None, output_agg = 'avg', comments="", lr=1e-3, opt="Adam"
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        d_model = hidden_dim
        n_layer = num_layers
        vocab_size = vocab_sz
        self.dropout_rate = dropout_rate
        ssm_cfg = {}
        rms_norm = True
        residual_in_fp32 = True
        fused_add_norm = True
        initializer_cfg = None
        # pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": None, "dtype": None}
        self.lr = lr
        self.opt = opt
        self.comments = comments

        self.output_agg = output_agg
        # if vocab_size % pad_vocab_size_multiple != 0:
        #     vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=None,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        # self.linear = nn.Linear(hidden_dim, output_dim)
        self.mlp = nn.Linear(d_model, output_dim)

        if self.dropout_rate:
            self.dropout_layer = nn.Dropout(p=dropout_rate)
        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.loss = nn.MSELoss()
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, src_key_padding_mask, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        input_ids = input_ids.to(torch.int)

        hidden_states = self.backbone(input_ids, inference_params=inference_params)
        if self.dropout_rate:
            hidden_states = self.dropout_layer(hidden_states)
        output = self.mlp(hidden_states)

        output = output[:,:,0]
        keep_ind = (~src_key_padding_mask).to(torch.float)
        if self.output_agg=='avg':
            output = (output*keep_ind).sum(dim=-1) / keep_ind.sum(dim=-1) # average over time
        elif self.output_agg=='sum':
            output = (output*keep_ind).sum(dim=-1)
        elif self.output_agg=='last':
            pos = src_key_padding_mask.int().argmax(dim=1)
            output = output[torch.arange(output.size(0)), pos]
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
        if self.opt == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        src, tgt, mask = batch['ids'], batch['hl'], batch['mask']
        output = self(src, mask)
        return output