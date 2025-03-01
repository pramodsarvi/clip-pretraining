import torch
import torch.nn as nn
from models import * 
import numpy as np
 
class CLIP(nn.Module):
 
    def __init__(
        self,
        emb_dim,
        vit_layers,
        vit_d_model,
        img_size,
        patch_size,
        n_channels,
        vit_heads,
        vocab_size,
        max_seq_length,
        text_heads,
        text_layers,
        text_d_model,
        retrieval=False,
    ):
        super().__init__()
 
        self.vision_encoder = VisionEncoder(
            vit_d_model,
            img_size,
            patch_size,
            n_channels,
            vit_heads,
            vit_layers,
            emb_dim,
        )
        # print(retrieval)
        
        self.text_encoder = TextEncoder(
            vocab_size,
            text_d_model,
            max_seq_length,
            text_layers,
            text_heads,
            emb_dim,
        )
 
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def CLIPLoss(self, logits, device="cuda"):
        # Symmetric or Contrastive loss
        # arange generates a list between 0 and n-1
        labels = torch.arange(logits.shape[0]).to(
            device
        )  # For row 1 we want 1,1 to be max, and row n-1 we want (n-1,n-1) text pairs to be max --> time 15.43 umar
 
        loss_v = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
 
        loss_t = nn.functional.cross_entropy(logits, labels)
        loss = (loss_v + loss_t) / 2
 
        return loss
    

    def forward(self, image, text, mask=None):
        V_e = self.vision_encoder(image)  # Vision encoder output [B, emb_dim]
        T_e = self.text_encoder(text, mask)  # Text encoder output [B, emb_dim]
        # print(f"V_e shape: {V_e.shape}, T_e shape: {T_e.shape}")
 
   
        logits = (V_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)
         
        loss = self.CLIPLoss(logits, self.device)
         
        return loss
    
