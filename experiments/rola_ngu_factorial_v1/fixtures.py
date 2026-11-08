"""Random, <1M-parameter CPU fixtures. No downloads or pretrained assets."""
from types import SimpleNamespace
import torch
from torch import nn


class Norm(nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim,dtype=dtype))

    def forward(self,x):
        return x * torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6) * self.weight


class MLP(nn.Module):
    def __init__(self,dtype):
        super().__init__()
        self.gate_proj = nn.Linear(24,40,bias=False,dtype=dtype)
        self.up_proj = nn.Linear(24,40,bias=False,dtype=dtype)
        self.down_proj = nn.Linear(40,24,bias=False,dtype=dtype)

    def forward(self,x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))


def tiny(seed: int = 43, *, dtype=torch.float32):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = nn.Module(); m._rng_fixture = True
        m.config = SimpleNamespace(model_type='qwen3',hidden_size=24,head_dim=8,
            num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=3,
            vocab_size=8,max_position_embeddings=96,attention_dropout=0.0,
            use_sliding_window=False,rope_scaling=None,layer_types=['full_attention']*3)
        body = nn.Module(); body.embed_tokens=nn.Embedding(8,24,dtype=dtype)
        body.layers = nn.ModuleList()
        for _ in range(3):
            layer=nn.Module(); a=nn.Module()
            for name,inp,out in (('q_proj',24,32),('k_proj',24,16),('v_proj',24,16),('o_proj',32,24)):
                setattr(a,name,nn.Linear(inp,out,bias=False,dtype=dtype))
            a.q_norm,a.k_norm,a.scaling=Norm(8,dtype),Norm(8,dtype),8**-.5
            layer.self_attn=a
            layer.input_layernorm,layer.post_attention_layernorm=Norm(24,dtype),Norm(24,dtype)
            layer.mlp=MLP(dtype); body.layers.append(layer)
        body.norm=Norm(24,dtype); body.rotary_emb=nn.Module()
        body.rotary_emb.register_buffer('inv_freq',1/(10000**(torch.arange(0,8,2,dtype=dtype)/8)))
        body.rotary_emb.attention_scaling=1.0
        m.model=body; m.lm_head=nn.Linear(24,8,bias=False,dtype=dtype)
        return m


def independent_logits(model, ids, origin=0):
    """Separate dense formula, matrix rotations, head loops; no Actor helpers."""
    x=model.model.embed_tokens(ids); t=ids.shape[1]; d=model.config.head_dim
    pos=torch.arange(origin,origin+t,dtype=x.dtype,device=x.device)
    angle=pos[:,None]*model.model.rotary_emb.inv_freq[None,:]
    rotations=x.new_zeros(t,d,d)
    for j in range(d//2):
        rotations[:,j,j]=angle[:,j].cos(); rotations[:,j+d//2,j+d//2]=angle[:,j].cos()
        rotations[:,j,j+d//2]=-angle[:,j].sin(); rotations[:,j+d//2,j]=angle[:,j].sin()
    mask=torch.tensor([[j<=i for j in range(t)] for i in range(t)],device=x.device)
    for layer in model.model.layers:
        a=layer.self_attn; z=layer.input_layernorm(x)
        q=a.q_norm(a.q_proj(z).reshape(1,t,4,8)).transpose(1,2)
        k=a.k_norm(a.k_proj(z).reshape(1,t,2,8)).transpose(1,2)
        v=a.v_proj(z).reshape(1,t,2,8).transpose(1,2)
        q=torch.einsum('tij,bhtj->bhti',rotations,q); k=torch.einsum('tij,bhtj->bhti',rotations,k)
        heads=[]
        for h in range(4):
            s=(q[:,h]@k[:,h//2].transpose(-1,-2))/(8**.5)
            heads.append(s.masked_fill(~mask,-torch.inf).softmax(-1)@v[:,h//2])
        out=torch.stack(heads,1).transpose(1,2).reshape(1,t,32)
        x=x+a.o_proj(out); x=x+layer.mlp(layer.post_attention_layernorm(x))
    return model.lm_head(model.model.norm(x))
