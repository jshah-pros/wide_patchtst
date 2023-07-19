import torch
from model.patchtst import WidePatchTST

# Model parameters
config = {
    'enc_in': 512,
    'seq_len': 7,
    'pred_len': 1,
    'patch_len': 7,
    'stride': 1,
    'padding_patch': None,
    'e_layers': 2,
    'n_heads': 8,
    'd_model': 512,
    'd_ff': 2048,
    'd_cat': 12,
    'dropout': 0.05,
    'fc_dropout': 0.05,
    'head_dropout': 0.0,
    'individual': 0,
}       

# Initialize Model
patchTST = WidePatchTST(config)

# Test
x_ts = torch.rand(128, 7, 1)
x_cat = torch.rand(128, 12, 1)
y = torch.rand(128, 1, 1)
y_hat = patchTST(x_ts, x_cat)

print(y_hat.shape) # [BS, Output Length, Channels]
assert y.shape == y_hat.shape