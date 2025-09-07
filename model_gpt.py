import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class GPTModelArgs:
    # hyperparameters for GPT model
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50257  # GPT-2 vocab size
    hidden_dim: Optional[int] = None
    max_seq_len: int = 1024
    norm_eps: float = 1e-5
    dropout: float = 0.1
    tie_weights: bool = True  # tie input and output embeddings

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias

def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class GPTAttention(nn.Module):
    def __init__(self, args: GPTModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.max_seq_len = args.max_seq_len
        
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        
        self.dropout = nn.Dropout(args.dropout)
        
        # casual mask for inference
        self.register_buffer("mask", torch.tril(torch.ones(args.max_seq_len, args.max_seq_len)))
        
        # flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, layer_past=None):
        batch_size, seq_len, _ = x.shape
        
        # project queries, keys, values
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # handle past key/values for generation
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        present = (k, v) if layer_past is not None else None
        
        # flash attention implementation
        if self.flash and seq_len > 1:
            output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout.p if self.training else 0.0, 
                is_causal=True
            )
        else:
            # manual attention implementation
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # apply causal mask
            mask = self.mask[:seq_len, :seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # weighted sum of values
            output = torch.matmul(attn_weights, v)
        
        # restore original shape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        output = self.dropout(output)
        
        return output, present

class GPTFeedForward(nn.Module):
    def __init__(self, args: GPTModelArgs):
        super().__init__()
        hidden_dim = args.hidden_dim or 4 * args.dim
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = self.w1(x)
        x = gelu(x)  # GPT uses GELU, not SwiGLU like llama2.
        x = self.w2(x)
        return self.dropout(x)

class GPTBlock(nn.Module):
    def __init__(self, layer_id: int, args: GPTModelArgs):
        super().__init__()
        self.layer_id = layer_id
        
        self.attention_norm = LayerNorm(args.dim, eps=args.norm_eps)
        self.attention = GPTAttention(args)
        
        self.ffn_norm = LayerNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = GPTFeedForward(args)

    def forward(self, x, layer_past=None):
        # attention
        attn_output, present = self.attention(self.attention_norm(x), layer_past)
        x = x + attn_output
        
        # feed forward
        ffn_output = self.feed_forward(self.ffn_norm(x))
        x = x + ffn_output
        
        return x, present

class GPT(nn.Module):
    def __init__(self, args: GPTModelArgs):
        super().__init__()
        self.args = args
        
        # token and positional embeddings
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        
        # transformer layers
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(GPTBlock(layer_id, args))
        
        # final normalization and output
        self.norm = LayerNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # weight tying
        if args.tie_weights:
            self.output.weight = self.tok_embeddings.weight
        
        self.dropout = nn.Dropout(args.dropout)
        
        # initialize weights
        self.apply(self._init_weights)
        
        # for tracking loss
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None, past_key_values=None):
        batch_size, seq_len = tokens.shape
        
        # token embeddings
        tok_emb = self.tok_embeddings(tokens)
        
        # positional embeddings
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        pos_emb = self.pos_embeddings(positions)
        
        # combine embeddings
        x = self.dropout(tok_emb + pos_emb)
        
        # process through layers
        presents = []
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, layer_past)
            presents.append(present)
        
        x = self.norm(x)
        
        # calculate loss or return logits
        if targets is not None:
            logits = self.output(x)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(x[:, [-1], :])
            self.last_loss = None
        
        return logits, presents

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # use fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else {}
        
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # crop sequence if too long
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx