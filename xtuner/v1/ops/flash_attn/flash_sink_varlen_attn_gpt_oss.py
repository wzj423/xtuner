# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# This file contains code originally written by Wenhao Li.
# --------------------------------------------------------

import math
import torch
from lmdeploy.pytorch.kernels.cuda.flashattention import flash_attn_varlen_func

class FlashSinkVarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor,
        cu_seqlen: torch.Tensor,
        window_size=None,
    ):
        # Determine max sequence length for q from cu_seqlen
        max_seqlen_q = (cu_seqlen[1:] - cu_seqlen[:-1]).max().item()

        # Handle window_size
        if window_size == -1 or window_size is None:
            win_size = (-1, -1)
        else:
            win_size = window_size

        # Use lmdeploy's flash_attn_varlen_func
        # xtuner's k and v are (seqlen, num_kv_heads, head_dim), so kv_layout is 'shd'
        o = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlen,
            cu_seqlens_k=cu_seqlen,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=None,  # Defaults to 1/sqrt(d) in the kernel
            causal=True,
            window_size=win_size,
            softcap=0.0,
            sinks=sink,
            kv_layout='shd'
        )

        # The lmdeploy kernel does not return LSE, so we return None.
        # This implies backward pass is not supported.
        lse = None

        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        raise NotImplementedError("Backward pass is not supported with lmdeploy kernel backend.")

flash_sink_attn_varlen_func = FlashSinkVarlenAttention.apply
