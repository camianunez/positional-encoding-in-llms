"""Tiny pre-norm causal decoder with controlled position handling."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .positional import causal_bias, rotate, sinusoidal

METHODS = ("none", "learned", "sinusoidal", "rope", "single_rope", "alibi")

@dataclass
class ModelConfig:
    method: str = "rope"
    vocab: int = 84
    dim: int = 64
    heads: int = 4
    layers: int = 2
    max_positions: int = 256
    base: float = 10000.0

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.dim, 3*cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, positions, past=None, scale=1.0):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b,n,3,self.cfg.heads,d//self.cfg.heads).permute(2,0,3,1,4).unbind(0)
        if self.cfg.method in ("rope", "single_rope"):
            kw = dict(base=self.cfg.base, scale=scale, single=self.cfg.method=="single_rope")
            q, k = rotate(q, positions, **kw), rotate(k, positions, **kw)
        # Cache already-rotated keys. Existing keys are never rotated again.
        if past is not None:
            pk, pv, pp = past
            k, v = torch.cat((pk,k), dim=-2), torch.cat((pv,v), dim=-2)
            kpos = torch.cat((pp,positions))
        else:
            kpos = positions
        mask = causal_bias(positions,kpos,self.cfg.heads,alibi=self.cfg.method=="alibi",device=x.device)
        y = F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
        y = y.transpose(1,2).contiguous().view(b,n,d)
        return self.proj(y), (k,v,kpos)

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim,4*cfg.dim),nn.GELU(),nn.Linear(4*cfg.dim,cfg.dim))
    def forward(self,x,positions,past=None,scale=1.0):
        y,cache = self.attn(self.ln1(x),positions,past,scale)
        x = x+y
        return x+self.mlp(self.ln2(x)),cache

class TinyDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        if cfg.method not in METHODS:
            raise ValueError(f"unknown method {cfg.method}")
        if cfg.dim % cfg.heads or (cfg.dim//cfg.heads) % 2:
            raise ValueError("each head must have an even dimension")
        self.cfg = cfg
        # Initialize common modules before the optional table: matched seeds
        # start all shared parameters identically for every method.
        self.embed = nn.Embedding(cfg.vocab,cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim,cfg.vocab,bias=False)
        self.apply(self._init)
        self.position = nn.Embedding(cfg.max_positions,cfg.dim) if cfg.method=="learned" else None
        if self.position is not None:
            nn.init.normal_(self.position.weight,std=0.02)
    @staticmethod
    def _init(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=0.02)
            if isinstance(module,nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
    def forward(self,tokens,*,offset=0,past=None,scale=1.0):
        positions = torch.arange(offset,offset+tokens.shape[1],device=tokens.device)
        x = self.embed(tokens)
        if self.position is not None:
            x = x+self.position(positions)
        elif self.cfg.method=="sinusoidal":
            x = x+sinusoidal(positions,self.cfg.dim).to(x.dtype)
        caches = []
        for i,block in enumerate(self.blocks):
            x,cache = block(x,positions,None if past is None else past[i],scale)
            caches.append(cache)
        return self.output(self.norm(x)),caches
    def configuration(self):
        return asdict(self.cfg)
