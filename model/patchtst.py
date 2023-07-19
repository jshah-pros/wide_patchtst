from typing import Optional

import torch
from torch import nn, Tensor
from .layers import PatchTST_backbone


class WidePatchTST(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=24, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        context_window = configs['seq_len']
        target_window = configs['pred_len']
        
        n_layers = configs['e_layers']
        n_heads = configs['n_heads']
        d_model = configs['d_model']
        d_ff = configs['d_ff']
        d_cat = configs['d_cat']                # This parameter is added in WidePatchTST
        dropout = configs['dropout']
        fc_dropout = configs['fc_dropout']
        head_dropout = configs['head_dropout']
    
        patch_len = configs['patch_len']
        stride = configs['stride']
        
        # model
        self.model = PatchTST_backbone(context_window=context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, d_cat=d_cat, 
                                  norm=norm,attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe,
                                  fc_dropout=fc_dropout, head_dropout=head_dropout, verbose=verbose, **kwargs)
    
    
    def forward(self, x_ts, x_cat):   # x_ts: [Batch, Input length, Channel], x_cat: [Batch, ohe feats. dim, 1]
        x_ts = x_ts.permute(0,2,1)    # x_ts: [Batch, Channel, Input length]
        x_cat = x_cat.permute(0,2,1)  # x_cat: [Batch, 1, ohe feats. dim]
        y = self.model(x_ts, x_cat)   # y: [Batch, Channel, Output Length]
        y = y.permute(0,2,1)          # y: [Batch, Output length, Channel]
        
        return y