from typing import Optional, Tuple
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from einops.layers.torch import Rearrange

from utils import create_win_index

#flash attention turing api
try:
    from flash_attn_turing_api.flash_attention_interface import flash_attn_func, flash_attn_varlen_func
    from flash_attn_turing_api.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    raise ImportError('******* flash_attn_turing_api is not installed ******* ')
    
class VocabEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._weight = None
    
        self.weight = Parameter(torch.empty(self.num_embeddings, self.embedding_dim))
        init.xavier_normal_(self.weight)
        
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = F.embedding(
            input_,
            self.weight,
        )
        return output
 
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def apply_scaling(freqs: torch.Tensor):
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192 

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    if freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1]):
        return freqs_cis.unsqueeze(2)
    elif freqs_cis.shape == (x.shape[1], x.shape[-1]):
        return freqs_cis.view(1, x.shape[1], 1, x.shape[-1])
    else:
        raise ValueError('freqs_cis shape error')
    
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

class FlashAttention(nn.Module):
    def __init__(self, dim, heads, kv_heads, dropout=0.0, max_batch_size=32, max_seq_len=2048):
        '''
        Grouped-Query Attention (GQA) 
        '''
        super().__init__()
        self.n_heads = heads
        self.n_kv_heads = heads if kv_heads is None else kv_heads
        self.n_rep = heads // self.n_kv_heads
        self.head_dim = dim // heads

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False) 
        self.wo = nn.Linear(self.n_heads * self.head_dim,dim, bias=False) 

        self.dropout = dropout if self.training else 0.0
        self.causal = True

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        '''
        mask: attention_mask (bsz, cache_len + seqlen)
        '''
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        keys = xk
        values = xv

        assert xq.dtype in [torch.float16, torch.bfloat16]
        assert keys.dtype in [torch.float16, torch.bfloat16]

        output = self._flash_attention_forward(xq, keys, values, mask, seqlen, dropout=self.dropout)
        output = output.reshape(bsz, seqlen, -1).contiguous()
        output = self.wo(output)
        return output

    def _flash_attention_forward(self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim) 
        """
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                softmax_scale=softmax_scale,
                causal=self.causal,
            )
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(query_states, key_states, value_states, softmax_scale, self.causal)

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            ) 
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

class PixFlashAttention(FlashAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        '''
        x: (bsz, seqlen, d), bsz is _bwins
        mask: attention_mask (bsz, seqlen)
        freqs_cis: (b, wins*win_es, head_dim/2)
        '''
        bsz, seqlen, _ = x.shape
        win_es = seqlen
        wins_win_es = freqs_cis.size(1) 

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        _b = bsz//(wins_win_es//win_es)
        xq = xq.reshape(_b, wins_win_es, self.n_heads, self.head_dim)
        xk = xk.reshape(_b, wins_win_es, self.n_kv_heads, self.head_dim)
 
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        xq = xq.reshape(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        keys = xk
        values = xv

        assert xq.dtype in [torch.float16, torch.bfloat16]
        assert keys.dtype in [torch.float16, torch.bfloat16]

        output = self._flash_attention_forward(xq, keys, values, mask, seqlen, dropout=self.dropout)
        output = output.reshape(bsz, seqlen, -1).contiguous()
        output = self.wo(output)
        return output

class FeedForward(nn.Module):
    def __init__(self,dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float], dropout = 0.0):
        '''
        hidden_dim: hidden dim of MLP, that is (dim, hidden_dim) @ (hidden_dim, dim)
        '''
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False) 
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) 
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim, heads, kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, 
                 dropout=0.0, max_batch_size=32, max_seq_len=2048, use_pix_atten=False):
        super().__init__()
        self.n_heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        if use_pix_atten:
            self.attention = PixFlashAttention(dim=dim, heads=heads, kv_heads=kv_heads, 
                dropout=dropout, max_batch_size=max_batch_size, max_seq_len=max_seq_len) 
        else:
            self.attention = FlashAttention(dim=dim, heads=heads, kv_heads=kv_heads, 
                dropout=dropout, max_batch_size=max_batch_size, max_seq_len=max_seq_len) 
        self.feed_forward = FeedForward(
            dim = dim,
            hidden_dim = 3 * dim,
            multiple_of = multiple_of,
            ffn_dim_multiplier = ffn_dim_multiplier,
            dropout = dropout
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class PixEmbeding(nn.Module):
    def __init__(self, fold_factor=16, embedding_dim=2048):
        super().__init__()
        if fold_factor not in [1,2,4,8,16,32,64,128,256]:
            raise ValueError("fold factor wrong!")
        color_length = (256//fold_factor)
        tokens_length = color_length**3
        self.weight = Parameter(torch.empty(tokens_length + 1, embedding_dim)) 
        self.rearr = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1)
        self.index = torch.LongTensor([1, color_length, color_length**2])
        
        init.xavier_normal_(self.weight)
        
    def forward(self, x:torch.tensor):
        '''
        x: (bsz, 3, H, W)
        out: (b,H*W,d)
        '''
        if self.index.device != x.device:
            self.index = self.index.to(x.device)
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
        x = self.rearr(x)
        x = x * self.index
        x = x.sum(dim=2)
        x = x.detach()
        output = F.embedding(x, self.weight)
        return output

class GToL(nn.Module):
    def __init__(self, image_size, window_size):
        super().__init__()
        self.index, self.wins, self.win_es = create_win_index(image_size=image_size, kernel_size=window_size, step=window_size)
     
    def forward(self, x):
        '''
        x: (bsz, n, d)
        o: (bsz*wins, win_es, d)
        '''
        if self.index.device != x.device:
            self.index = self.index.to(x.device)
        d = x.size(-1) 
        x = torch.index_select(x, dim=1, index=self.index)     
        x = x.reshape(-1, self.win_es, d)
        return x
       
class ImageBlock(nn.Module):
    def __init__(self, word_dim, dim, heads, kv_heads, layers, 
        norm_eps, multiple_of, ffn_dim_multiplier, rope_theta, 
        use_scaled_rope, dropout, 
        fold_factor, image_size, window_size):
        '''
        window_size: local window size
        '''
        super().__init__()
        max_seq_len = window_size
        self.pix_emb = PixEmbeding(fold_factor=fold_factor, embedding_dim=dim)
        self.gtol = GToL(image_size=image_size, window_size=window_size)
        self.n_layers = layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(layers):
            self.layers.append(TransformerBlock(layer_id=layer_id, dim=dim, heads=heads,
                kv_heads=kv_heads, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier,
                norm_eps=norm_eps, dropout=dropout, max_seq_len=max_seq_len, 
                use_pix_atten=True))
        self.norm = RMSNorm(dim, eps=norm_eps)
        total_emb_length = self.gtol.wins * self.gtol.win_es
        self.freqs_cis = precompute_freqs_cis(dim // heads, total_emb_length, rope_theta, use_scaled_rope)
   
    def select_freqs_cis(self, freqs_cis, freqs_mask):
        '''
        freqs_mask (b, wins * win_es) 0 or 1
        '''
        cum_mask = torch.cumsum(freqs_mask, dim=1, dtype=torch.int64)
        cum_mask = cum_mask - 1 
        cum_mask = cum_mask.to(freqs_cis.device)
        sel_freqs_cis = freqs_cis[cum_mask]
        sel_freqs_cis = sel_freqs_cis.detach()
        return sel_freqs_cis
    
    def forward(self, image, mask_token, mask_window, last_token):
        '''
        img: (b, 3, H, W), H*W = wins * win_es
        mask_token: (b, wins)
        mask_window: (b, win_es)
        last_token: (b, wins)
        return: (b, wins, dim)
        '''
        h = self.pix_emb(image) 
        _b, wins_win_es = h.size(0), h.size(1)

        h = self.gtol(h)
        
        _bwins, win_es, d = h.size(0), h.size(1), h.size(2)
        wins = wins_win_es // win_es

        mask = torch.full((_b, wins, win_es), 1, device=h.device)
        mask_window = mask_window.unsqueeze(1)
        mask_window = torch.repeat_interleave(mask_window, wins, dim=1)
        last_token_select = (last_token != -1)
        mask[last_token_select] = mask_window[last_token_select]
        freqs_mask =  mask * mask_token.unsqueeze(-1)
        freqs_mask = freqs_mask.reshape(_b, -1)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.select_freqs_cis(self.freqs_cis, freqs_mask)
        mask = mask.reshape(-1, win_es)

        for layer in self.layers:
            h = layer(h, 0, freqs_cis, mask)

        h = h.reshape(_b, wins, win_es, d)
        idx_b = torch.arange(_b, dtype=torch.long, device=h.device).view(-1, 1)
        idx_wins = torch.arange(wins, dtype=torch.long, device=h.device).view(1, -1)
        last_token = last_token.to(h.device)
        output = h[idx_b, idx_wins, last_token]

        return output

class UPW(nn.Module):
    '''
    Paper: Unified Pix Token And Word Token Generative Language Model
    This model is support for mixed text and image pretraining experiment.
    '''
    def __init__(self, dim, heads, kv_heads, layers, vocab_size, 
        norm_eps, multiple_of, ffn_dim_multiplier, rope_theta, 
        use_scaled_rope, dropout, max_batch_size, max_seq_len,
        image_layers, fold_factor, image_size, window_size, 
        PAD_TOKEN_ID=3, IMAGE_START_TOKEN_ID=4, IMAGE_END_TOKEN_ID=5,
        train_with_ddp=True):
        '''
        dim: word token embeding dim
        heads: Query heads number
        kv_heads: Key Value heads number
        layers: layers of decoders in text block
        vocab_size: word token total
        norm_eps: RMSNorm eps
        multiple_of: for MLP in decoder layer
        ffn_dim_multiplier: for MLP in decoder layer
        rope_theta: for position emb
        use_scaled_rope: for position emb
        dropout: for attention droupout
        max_batch_size: for kv cache
        max_seq_len: for position emb and kv cache
        image_layers: layers of decoders in image block
        fold_factor: folding factor
        image_size: image size
        window_size: local window size
        train_with_ddp: train by pytorch DDP
        '''
        super().__init__()
        self.PAD_TOKEN_ID = PAD_TOKEN_ID
        self.IMAGE_START_TOKEN_ID = IMAGE_START_TOKEN_ID
        self.IMAGE_END_TOKEN_ID = IMAGE_END_TOKEN_ID
        self.pad_id_tensor = None
        self.image_start_id_tensor = None
        self.n_layers = layers
        self.train_with_ddp = train_with_ddp
  
        self.tok_embeddings = VocabEmbedding(vocab_size, dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(layers):
            self.layers.append(TransformerBlock(layer_id=layer_id, dim=dim, heads=heads,
                kv_heads=kv_heads, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier,
                norm_eps=norm_eps, dropout=dropout, max_batch_size=max_batch_size, max_seq_len=max_seq_len,
                use_pix_atten=False))

        self.norm = RMSNorm(dim, eps=norm_eps)

        color_length = 256//fold_factor
        color_length = color_length**3
        self.output = nn.Linear(dim, vocab_size + color_length, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            dim // heads,
            max_seq_len * 2,
            rope_theta,
            use_scaled_rope,
        )

        #image block
        self.imgblk = ImageBlock(word_dim=dim, dim=dim, heads=heads, kv_heads=kv_heads, layers=image_layers, 
            norm_eps=norm_eps, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier, 
            rope_theta=rope_theta, use_scaled_rope=use_scaled_rope, dropout=dropout, 
            fold_factor=fold_factor, image_size=image_size, window_size=window_size)

    def select_position_emb(self, freqs_cis, mask):
        '''
        select correct postion emb for pad token
        mask: (bsz, seq_len)
        '''
        cum_mask = torch.cumsum(mask, dim=1, dtype=torch.int64)
        cum_mask = cum_mask - 1 
        cum_mask = cum_mask.to(freqs_cis.device)
        sel_freqs_cis = freqs_cis[cum_mask]
        sel_freqs_cis = sel_freqs_cis.detach()
        return sel_freqs_cis
    
    def forward(self, datas_list):
        '''
        datas_list: List[ List[DataElement] ]
        '''
        # DDP 
        if self.train_with_ddp:
            batch_length = len(datas_list)
            batch_length = batch_length // torch.cuda.device_count()
            batch_index = self.output.weight.device.index
            batch_device = self.output.weight.device
            datas = datas_list[batch_index:(batch_index+1)*batch_length]
        else:
            datas = datas_list

        #get all image
        images = []
        mask_tokens = []
        last_tokens = []
        mask_windows = []
        for data in datas:
            for ele in data:
                if ele.role == 'image':
                    images.append(ele.image)
                    mask_tokens.append(ele.mask_token)
                    last_tokens.append(ele.last_token)
                    mask_windows.append(ele.mask_window)
        images = torch.stack(images).to(batch_device)
        mask_tokens = torch.stack(mask_tokens).to(batch_device)
        last_tokens = torch.stack(last_tokens).to(batch_device)
        mask_windows = torch.stack(mask_windows).to(batch_device)

        pts = self.imgblk(images, mask_tokens, mask_windows, last_tokens) 
        # pad token emb
        if self.pad_id_tensor==None:
            self.pad_id_tensor = torch.tensor(self.PAD_TOKEN_ID, dtype=torch.int32, device=batch_device)
        pad = self.tok_embeddings(self.pad_id_tensor)
        not_img_mask = ~(mask_tokens.bool())
        pts[not_img_mask] = pad

        #get all text
        image_index = 0
        emb_batch = []
        mask_batch = []
        for data in datas:
            emb_data = []
            mask_data = []
            for ele in data:
                if ele.role == 'text':
                    tokens = ele.tokens.to(batch_device)
                    #word embeding
                    emb_ = self.tok_embeddings(tokens)
                    emb_data.append(emb_)
                    #mask
                    mask_ = ~(tokens == self.PAD_TOKEN_ID)
                    mask_data.append(mask_)
                elif ele.role == 'image':
                    emb_data.append(pts[image_index])
                    #mask
                    mask_data.append(mask_tokens[image_index])
                    image_index += 1
            emb_data = torch.cat(emb_data, dim=0)
            emb_batch.append(emb_data)
            mask_data = torch.cat(mask_data)
            mask_batch.append(mask_data)
        emb_batch = torch.stack(emb_batch)
        mask_batch = torch.stack(mask_batch)

        self.freqs_cis = self.freqs_cis.to(batch_device)
        freqs_cis = self.select_position_emb(self.freqs_cis, mask_batch)
       
        h = emb_batch
        for layer in self.layers:
            h = layer(h, 0, freqs_cis, mask_batch)
        
        h = self.norm(h)

        h = self.output(h)

        return h