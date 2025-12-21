import torch
import torch.nn as nn
from torch.nn.init import (
    xavier_uniform_,
    constant_,
)
from torch.nn.functional import linear
from einops import rearrange


def _in_projection_packed(q, k, v, w, b=None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Scaled dot-product attention accelerated by flash-attn if available.
    Fallback to standard attention on CPU without dependency.
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        try:
            from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func  # noqa: F401
            from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis  # noqa: F401
            self._has_flash = True
        except Exception:
            self._has_flash = False

    def forward(self, q, kv, causal=False, key_padding_mask=None):
        if self._has_flash and q.is_cuda and kv.is_cuda and q.dtype in [torch.float16, torch.bfloat16]:
            from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
            from flash_attn.bert_padding import unpad_input
            batch_size = q.shape[0]
            seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
            if key_padding_mask is None:
                q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
                max_sq, max_sk = seqlen_q, seqlen_k
                cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                            device=q.device)
                cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                            device=kv.device)
                output = flash_attn_unpadded_kvpacked_func(
                    q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = kv.shape[-2]
                q = rearrange(q, 'b s ... -> (b s) ...')
                max_sq = seqlen_q
                cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                            device=q.device)
                x = rearrange(kv, 'b s two h d -> b s (two h d)')
                x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
                output_unpad = flash_attn_unpadded_kvpacked_func(
                    q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)
            return output, None

        # Fallback: standard attention
        q, k, v = q, kv[:, :, 0], kv[:, :, 1]
        scale = self.softmax_scale or (q.shape[-1] ** -0.5)
        attn = torch.einsum('bshd,bThd->bhst', q, k) * scale
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=attn.dtype)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = nn.functional.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.einsum('bhst,bThd->bshd', attn, v)
        return out, None


class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, key_padding_mask=None):
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


