import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads, drop_out_rate=0.1, block_size=500):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_head = heads
        self.head_dim = embed_dim // heads
        
        
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    
    def forward(self, x, mask=None, pad_token_mask=None):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # print(f"query shape===> {q.shape}\n mask shape: {mask.shape}")
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if pad_token_mask is not None:
            # print(f"pad mask cal,x shape{x.shape}, pad shape:{pad_token_mask.shape}")
            expanded_pad_token_mask = torch.ones((B, T), device=x.device)
            expanded_pad_token_mask[:pad_token_mask.shape[0], :pad_token_mask.shape[1]] = pad_token_mask
            expanded_pad_token_mask = expanded_pad_token_mask.unsqueeze(1).unsqueeze(2)
            # _pad_token_mask = pad_token_mask.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1)
            # print(f"mask shape{mask.shape}")
            # expanded_pad_token_mask[:pad_token_mask.shape[0], :pad_token_mask.shape[1], :pad_token_mask.shape[1]] = _pad_token_mask # Match the size and fill 1s where mask is 0
            att = att.masked_fill(expanded_pad_token_mask == 0, float('-inf'))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.fc_out(y))
        return y


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_head, drop_out_rate=0.1, fc_rate=4):
        super(TransformerDecoderLayer, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )
    
    def forward(self, x,  pad_token_mask=None):
        x = x + self.attn(self.ln1(x), pad_token_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 embed_size, 
                 num_layers, 
                 heads,
                 dropout):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size
        # self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_size, heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        # self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, out, pad_token_mask=None):
        # N, seq_length = x.shape
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        # out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # print(f"input==========++> {out}")
        for layer in self.layers:
            out = layer(out, pad_token_mask)
            # print(f"layer out{out}")
        out = self.ln_f(out)
        # out = self.fc_out(out)
        
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, 
                 embed_size, 
                 num_layers, 
                 heads,
                 dropout):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        # self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(embed_size, heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        # self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, out, pad_token_mask=None):
        # N, seq_length = x.shape
        # positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        # out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        # print(f"input==========++> {out}")
        for layer in self.layers:
            out = layer(out, pad_token_mask)
            # print(f"layer out{out}")
        # out = self.ln_f(out)
        # out = self.fc_out(out)
        
        return out