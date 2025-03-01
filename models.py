import torch
import torch.nn as nn
import torchvision.transforms as T
 
import numpy as np
 

class AttentionHead(nn.Module):
    def __init__(self, d_model, qkv_dim):
        super().__init__()
 
        self.qkv_dim = qkv_dim
 
        self.query = nn.Linear(d_model, qkv_dim)
        self.key = nn.Linear(d_model, qkv_dim)
        self.value = nn.Linear(d_model, qkv_dim)
 
    def forward(self, x, mask=None):
        # x.shape -->  [B,max_seq_len,d_model]
        Q = self.query(x)  # [B,max_seq_len,vit_heads]
        K = self.key(x)
        V = self.value(x)
 
        attention = Q @ K.transpose(
            -2, -1
        )  # eg: -2 -second last dim and -1 last dim -->  [B,max_seq_len,max_seq_len]
        # Scaling
        attention = attention / self.qkv_dim**0.5  #  [B,max_seq_len,max_seq_len]
 
        # Apply attention mask for padded sequence
        if mask is not None:
            mask = attention.masked_fill(
                mask == 0, float("-inf")
            )  # torch.tensor.masked_fill
 
        # Apply softmax to obtain attention weights [Wij]
        attention = torch.softmax(
            attention, dim=-1
        )  # along last dim  # (softmax(Q_K^T)/sqrt(d_k)).V -->  [B,max_seq_len,max_seq_len]
 
        attention = attention @ V  #  [B,max_seq_len,max_seq_len]
 
        return attention  # Y_i
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
 
        # d_model --> embed dimension
        # n_heads --> number of heads
        self.qkv_dim = d_model // n_heads  # or self.head_size
 
        self.W_o = nn.Linear(d_model, d_model)  # Dense layer
 
        self.multi_head = nn.ModuleList(
            [AttentionHead(d_model, self.qkv_dim) for _ in range(n_heads)]
        )
 
    def forward(self, x, mask=None):
 
        # x.shape --> [B,max_seq_len,d_model]
 
        # Concatenates the outputs from all attention heads along the last dimension (dim=-1)
 
        out = torch.cat(
            [head(x, mask=mask) for head in self.multi_head], dim=-1
        )  #  [B,max_seq_len,d_model]
 
        # Apply the linear transformation
        out = self.W_o(out)  # (Concat --> Dense)  --> [B,max_seq_len,d_model]
 
        return out

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, mlp_ratio =4):
      super().__init__()
 
      self.d_model = d_model
      self.n_heads = n_heads
 
      self.ln1 = nn.LayerNorm(d_model)
 
      self.mha = MultiheadAttention(d_model, n_heads)
 
      self.ln2 = nn.LayerNorm(d_model)
 
      self.mlp = nn.Sequential(
          nn.Linear(d_model, d_model*mlp_ratio),
          nn.GELU(),
          nn.Linear(d_model * mlp_ratio, d_model)
      )
 
#For clip even though its a encoder model it requires mask ->to account for padded for max seq_length
  def forward(self, x, mask = None):
 
      x_n = self.mha(self.ln1(x), mask = mask)
      x = x + self.mlp(self.ln2(x_n))
 
      return x  # x.shape -->  [B,max_seq_len,d_model]
  

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
 
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
 
        self.register_buffer("pe", pe.unsqueeze(0))
 
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
    



class VisionEncoder(nn.Module):
 
    def __init__(
        self, d_model, img_size, patch_size, n_channels, n_heads, n_layers, emb_dim
    ):
        super().__init__()
 
        assert (
            img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        ), "image dimensions should be divisible by patch dim"
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"
 
        self.num_patches = (img_size[0] * img_size[1]) // (
            patch_size[0] * patch_size[1]
        )  # max_seq_length
 
        self.max_seq_length = self.num_patches + 1
 
        self.linear_proj = nn.Conv2d(
            in_channels=n_channels,
            out_channels=d_model,
            kernel_size=patch_size[0],
            stride=patch_size[0],
        )
 
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True)
 
        self.positional_embedding = PositionalEmbedding(d_model, self.max_seq_length)
 
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]
        )
 
        self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
 
    def forward(self, x, mask=None):
 
        x = self.linear_proj(
            x
        )  # (B, C, H, W) -> (B, d_model, Patch_col_d_model, Patch_row_height)
 
        x = x.flatten(2).transpose(
            -2, -1
        )  # (B, d_model, Patch_col_d_model, Patch_row_height) --> Flatten (B, d_model, Patch) --> .transpose(-2,-1) (B, Patch, d_model)
 
        # The input to the transformer we need to pass a sequence of patches or tokens so we need num_patches to be before hidden dim
 
        x = torch.cat(
            (self.cls_token.expand(x.shape[0], -1, -1), x), dim=1
        )  # add cls token at the beginning of patch_sequence   -->  [B,max_seq_len,d_model]
 
        x = self.positional_embedding(x)  #  [B,max_seq_len,d_model]

        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x, mask)  # [B, d_model]
 
       # Get learned class tokens
        x = x[:, 0, :] 
       # Project to shared embedding space
        if self.projection is not None:
            x  = x  @ self.projection  # [B, emb_dim]
        x  = x  / torch.norm(x , dim = -1 , keepdim = True)
    
        return x
    

def tokenizer(text, encode=True, mask=None, max_seq_length=32):
    if encode:
        # Adding SOT and EOT tokens
        out = chr(2) + text + chr(3)
 
        # Truncate if length exceeds max_seq_length
        if len(out) > max_seq_length:
            out = out[:max_seq_length]
 
        # Add padding if needed
        out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])
 
        # Encode the text
        out = torch.IntTensor(list(out.encode("utf-8")))
 
        # Create the mask
        mask = torch.ones(len(out.nonzero()))
 
        # Pad the mask to max_seq_length
        if len(mask) < max_seq_length:
            mask = torch.cat((mask, torch.zeros(max_seq_length - len(mask)))).type(
                torch.IntTensor
            )
        else:
            mask = mask.type(torch.IntTensor)
    else:
        # Decode the text
        out = [chr(x) for x in text[1 : len(mask.nonzero()) - 1]]
        out = "".join(out)
        mask = None
 
    return out, mask



class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length, n_layers, n_heads, emb_dim):
        super().__init__()
 
        self.max_seq_length = max_seq_length
 
        self.embed = nn.Embedding(vocab_size, d_model)
 
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_length)
 
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoder(d_model, n_heads) for _ in range(n_layers)]
        )
 
        self.projection = nn.Parameter(torch.randn(d_model, emb_dim))
 
    # For training
    def forward(self, text, mask=None):
 
        x = self.embed(text)
 
        x = self.positional_embedding(x)
 
        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x, mask=mask)
 
        # The output of the encoder layers is the text features. We are going to be using the features from the EOT embedding.
 
        x = x[torch.arange(text.shape[0]), torch.sub(torch.sum(mask[:, 0], dim=1), 1)]
 
        if self.projection is not None:
            x = x @ self.projection
 
        x = x / torch.norm(x, dim=-1, keepdim=True)
 
        return x
    
