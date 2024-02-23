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



class MixerModelWithEmbeddingInput(MixerModel):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        ignore_input_ids=False
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        if not ignore_input_ids:
            self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        self.ignore_input_ids = ignore_input_ids            
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, input_ids, embs, inference_params=None):
        if not self.ignore_ids:
            hidden_states = self.embedding(input_ids) + embs
        else:
            hidden_states = embs

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states
        


class MambaSingleOutputModelWithEmbeddingInput(pl.LightningModule, GenerationMixin):

    def __init__(
        self,
        vocab_sz, output_dim, hidden_dim, num_layers, input_emb_dim, dropout_rate = None, ignore_ids = False, comments="", lr=1e-3, opt="Adam"
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
        self.input_emb_dim = input_emb_dim
        self.ignore_ids = ignore_ids
        # if vocab_size % pad_vocab_size_multiple != 0:
        #     vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModelWithEmbeddingInput(
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
        self.mlp_output = nn.Linear(d_model, output_dim)
        self.mlp_input = nn.Linear(input_emb_dim, d_model)
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

    def forward(self, input_ids, embs, src_key_padding_mask, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        input_ids = input_ids.to(torch.int)
        hidden_states = self.backbone(input_ids, self.mlp_input(embs.float()), inference_params=inference_params)
        if self.dropout_rate:
            hidden_states = self.dropout_layer(hidden_states)
        output = self.mlp_output(hidden_states)

        output = output[:,:,0]
        keep_ind = (~src_key_padding_mask).to(torch.float)
        output = (output*keep_ind).sum(dim=-1) / keep_ind.sum(dim=-1) # average over time

        return output

    

    def training_step(self, batch, batch_idx):
        src, tgt, mask, embs = batch[0]['ids'], batch[0]['hl'], batch[0]['mask'], batch[1]['embs']
        # print(tgt.dtype)
        output = self(src, embs, mask)
        # print(output.dtype, output.shape)

        loss = self.loss(tgt, output)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt, mask, embs = batch[0]['ids'], batch[0]['hl'], batch[0]['mask'], batch[1]['embs']
        output = self(src, embs, mask)
        loss = self.loss(tgt, output)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        if self.opt == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer