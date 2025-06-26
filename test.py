import torch
import os
from collections import OrderedDict

# Load checkpoint
print("Loading checkpoint...")
checkpoint = torch.load("checkpoint.ckpt", map_location="cpu")

# Get model state dict
if "model_state" in checkpoint:
    model_state = checkpoint["model_state"]
    print("\nModel state keys:")
    for key, tensor in model_state.items():
        if isinstance(tensor, torch.Tensor):
            print(f"- {key}: {tensor.shape}")
        else:
            print(f"- {key}: Not a tensor (type: {type(tensor)})")
    
    # Save just the model weights for encoder-decoder
    torch.save(model_state, "encoder_decoder_weights.pth")
    print("\nSaved model weights to encoder_decoder_weights.pth")
else:
    print("\nError: Could not find model weights in checkpoint")