"""
Text2Semantic Module - Text to VQ code generation.
Used by tts.py for converting text to audio semantic codes.
"""
import base64
import dataclasses
import json
import math
import os
import platform
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import tiktoken
import torch
import torch._inductor.config
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm


log_info = lambda msg: print(f"[INFO] {msg}")
log_warning = lambda msg: print(f"[WARNING] {msg}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True


# ============= Tokenizer =============

FISH_TIKTOKEN_PATTERN = "|".join([
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"\p{P}",
    r"[^\r\n\p{L}\p{N}]?\p{L}+",
    r"\p{N}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(\?!\S)",
    r"\s+",
])
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

IM_END_TOKEN = "<|im_end|>"
MODALITY_TOKENS = {"text": "<|text|>", "voice": "<|voice|>", "interleave": "<|interleave|>"}
SEMANTIC_TOKENS = [f"<|semantic:{i}|>" for i in range(4096)]
ALL_SPECIAL_TOKENS = [
    "<|begin_of_text|>", "<|end_of_text|>", "<|pad|>", "<|im_start|>", "<|im_end|>",
    "<|phoneme_start|>", "<|phoneme_end|>", "<|tool_call_start|>", "<|tool_call_end|>",
    "<|text|>", "<|voice|>", "<|interleave|>", "<|audio_start|>", "<|audio_end|>", "<|audio|>",
    *SEMANTIC_TOKENS,
]


class FishTokenizer:
    def __init__(self, model_path: str, special_tokens: list = ALL_SPECIAL_TOKENS):
        mergeable_ranks = self.load_tiktoken_bpe(model_path)
        special_token_begin = len(mergeable_ranks)
        self.all_special_tokens_with_ids = {token: special_token_begin + i for i, token in enumerate(special_tokens)}

        self.semantic_id_to_token_id = {}
        end_idx = 0
        for token in special_tokens:
            if token.startswith("<|semantic:"):
                match = re.match(r"<\|semantic:(\d+)\|>", token)
                if match:
                    idx = int(match.group(1))
                    self.semantic_id_to_token_id[idx] = self.all_special_tokens_with_ids[token]
                    if idx > end_idx:
                        end_idx = idx

        self.semantic_begin_id = self.semantic_id_to_token_id[0]
        self.semantic_end_id = self.semantic_id_to_token_id[end_idx]

        self.tkt_model = tiktoken.core.Encoding(
            name=Path(model_path).stem,
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.all_special_tokens_with_ids,
        )

    @staticmethod
    def load_tiktoken_bpe(tiktoken_bpe_file: str):
        data = {}
        for line in open(tiktoken_bpe_file).read().splitlines():
            if not line:
                continue
            token, rank = line.split()
            if token == "=":
                continue
            data[base64.b64decode(token)] = int(rank)
        return data

    def get_token_id(self, token: str) -> int:
        return self.all_special_tokens_with_ids[token]

    def encode(self, s: str, allowed_special=True):
        subs = [s[i:i + TIKTOKEN_MAX_ENCODE_CHARS] for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)]
        if allowed_special is True:
            allowed_special = self.tkt_model.special_tokens_set
        elif allowed_special is False:
            allowed_special = set()
        return sum(self.tkt_model.encode_batch(subs, allowed_special=allowed_special, disallowed_special=set()), start=[])

    @staticmethod
    def from_pretrained(path: str):
        special_tokens_path = Path(path) / "special_tokens.json"
        if special_tokens_path.exists():
            with open(special_tokens_path) as f:
                all_special_tokens_with_ids = json.load(f)
        else:
            all_special_tokens_with_ids = ALL_SPECIAL_TOKENS
        return FishTokenizer(str(Path(path) / "tokenizer.tiktoken"), all_special_tokens_with_ids)


# ============= Content Sequence =============

@dataclass
class BasePart:
    type: Literal["text", "vq"] = None
    cal_loss: bool = False


@dataclass(kw_only=True)
class VQPart(BasePart):
    codes: torch.Tensor

    def __post_init__(self):
        self.type = "vq"
        if isinstance(self.codes, np.ndarray):
            self.codes = torch.from_numpy(self.codes.copy())


@dataclass(kw_only=True)
class TextPart(BasePart):
    text: str = None
    tokens: list = None

    def __post_init__(self):
        self.type = "text"


@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_mask_tokens: torch.Tensor = None
    vq_parts: list = None


@dataclass
class ContentSequence:
    parts: list = field(default_factory=list)
    modality: Literal["text", "voice", "interleave"] = None

    def __init__(self, parts=None, modality=None):
        self.modality = modality
        self.parts = []
        for part in parts or []:
            if isinstance(part, dict):
                if part["type"] == "vq":
                    part = VQPart(**part)
                elif part["type"] == "text":
                    part = TextPart(**part)
            self.parts.append(part)

        if self.modality:
            self.parts.insert(0, TextPart(text=MODALITY_TOKENS[self.modality]))

    def append(self, part_or_parts, add_end=False, speaker=None):
        parts_to_add = [part_or_parts] if not isinstance(part_or_parts, list) else part_or_parts
        if speaker is not None:
            self.parts.append(TextPart(text=f"<|speaker:{speaker}|>"))
        self.parts.extend(parts_to_add)
        if add_end:
            self.parts.append(TextPart(text=IM_END_TOKEN))

    def encode(self, tokenizer, add_shift=True):
        all_tokens, all_labels, vq_parts, vq_masks = [], [], [], []

        for part in self.parts:
            if isinstance(part, TextPart):
                tokens = tokenizer.encode(part.text) if part.tokens is None else part.tokens
                tokens = torch.tensor(tokens, dtype=torch.int)
            elif isinstance(part, VQPart):
                curr_codes = part.codes.clone().to(torch.int)
                tokens = torch.tensor([tokenizer.semantic_id_to_token_id[int(i.item())] for i in curr_codes[0].int()], dtype=torch.int)
                vq_parts.append(curr_codes)
            else:
                continue

            all_tokens.append(tokens)
            vq_masks.append(torch.ones_like(tokens, dtype=torch.bool) if isinstance(part, VQPart) else torch.zeros_like(tokens, dtype=torch.bool))
            all_labels.append(tokens.clone() if part.cal_loss else torch.full_like(tokens, -100))

        tokens = torch.cat(all_tokens, dim=0)
        labels = torch.cat(all_labels, dim=0)
        vq_masks = torch.cat(vq_masks, dim=0)

        vq_mask_tokens = vq_masks
        if add_shift:
            tokens = tokens[:-1]
            labels = labels[1:]
            vq_mask_tokens = vq_mask_tokens[:-1]

        return EncodedMessage(tokens=tokens, labels=labels, vq_parts=vq_parts, vq_mask_tokens=vq_mask_tokens)

    def encode_for_inference(self, tokenizer, num_codebooks):
        encoded = self.encode(tokenizer, add_shift=False)
        tokens = encoded.tokens
        values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int)
        values[0] = tokens

        if encoded.vq_parts and len(encoded.vq_parts) > 0:
            vq_parts = torch.cat(encoded.vq_parts, dim=1)
            values[0, encoded.vq_mask_tokens] = vq_parts[0] + tokenizer.semantic_begin_id
            values[1:, encoded.vq_mask_tokens] = vq_parts

        return values, None, None


# ============= Model =============

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class DualARModelArgs:
    model_type: str = "dual_ar"
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = False
    codebook_size: int = 160
    num_codebooks: int = 4
    use_gradient_checkpointing: bool = True
    initializer_range: float = 0.02
    is_reward_model: bool = False
    scale_codebook_embeddings: bool = False
    n_fast_layer: int = 4
    fast_dim: int = None
    fast_n_head: int = None
    fast_n_local_heads: int = None
    fast_head_dim: int = None
    fast_intermediate_size: int = None
    fast_attention_qkv_bias: bool = None
    fast_attention_qk_norm: bool = None
    fast_attention_o_bias: bool = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_head

        self.fast_dim = self.fast_dim or self.dim
        self.fast_n_head = self.fast_n_head or self.n_head
        self.fast_n_local_heads = self.fast_n_local_heads or self.n_local_heads
        self.fast_head_dim = self.fast_head_dim or self.head_dim
        self.fast_intermediate_size = self.fast_intermediate_size or self.intermediate_size
        self.fast_attention_qkv_bias = self.fast_attention_qkv_bias if self.fast_attention_qkv_bias is not None else self.attention_qkv_bias
        self.fast_attention_qk_norm = self.fast_attention_qk_norm if self.fast_attention_qk_norm is not None else self.attention_qk_norm
        self.fast_attention_o_bias = self.fast_attention_o_bias if self.fast_attention_o_bias is not None else self.attention_o_bias

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DualARModelArgs(**data)


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val
        return self.k_cache, self.v_cache


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = (x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + self.eps)).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], -1)
    return x_out2.flatten(3).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim, n_head, n_local_heads, head_dim, dropout=0.0, attention_qkv_bias=False, attention_o_bias=False, attention_qk_norm=False, norm_eps=1e-5, use_sdpa=True):
        super().__init__()
        total_head_dim = (n_head + 2 * n_local_heads) * head_dim
        self.wqkv = nn.Linear(dim, total_head_dim, bias=attention_qkv_bias)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=attention_o_bias)
        self.kv_cache = None

        if attention_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim, norm_eps)
            self.k_norm = nn.RMSNorm(head_dim, norm_eps)

        self.dropout = dropout
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_local_heads = n_local_heads
        self.use_sdpa = use_sdpa
        self.attention_qk_norm = attention_qk_norm

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.attention_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.use_sdpa:
            if mask is None:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        else:
            L, S = q.size(-2), k.size(-2)
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_bias = torch.zeros(1, 1, L, S, dtype=q.dtype, device=q.device)
            if mask is not None:
                if mask.dtype == torch.bool:
                    attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
                else:
                    attn_bias += mask
            attn_weight = q @ k.transpose(-2, -1) * scale_factor + attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            y = attn_weight @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, q_size)
        return self.wo(y)


class TransformerBlock(nn.Module):
    def __init__(self, config: DualARModelArgs, use_sdpa: bool = True):
        super().__init__()
        self.attention = Attention(config.dim, config.n_head, config.n_local_heads, config.head_dim, config.dropout, config.attention_qkv_bias, config.attention_o_bias, config.attention_qk_norm, config.norm_eps, use_sdpa)
        self.feed_forward = FeedForward(config.dim, config.intermediate_size)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        return h + self.feed_forward(self.ffn_norm(h))


class FastTransformerBlock(nn.Module):
    def __init__(self, config: DualARModelArgs, use_sdpa: bool = False):
        super().__init__()
        self.attention = Attention(config.fast_dim, config.fast_n_head, config.fast_n_local_heads, config.fast_head_dim, config.dropout, config.fast_attention_qkv_bias, config.fast_attention_o_bias, config.fast_attention_qk_norm, config.norm_eps, use_sdpa)
        self.feed_forward = FeedForward(config.fast_dim, config.fast_intermediate_size)
        self.ffn_norm = RMSNorm(config.fast_dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.fast_dim, config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        return h + self.feed_forward(self.ffn_norm(h))


class DualARTransformer(nn.Module):
    def __init__(self, config: DualARModelArgs, tokenizer: FishTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(config.codebook_size * config.num_codebooks, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if config.tie_word_embeddings is False:
            self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.register_buffer("freqs_cis", precompute_freqs_cis(config.max_seq_len, config.head_dim, config.rope_base), persistent=False)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)), persistent=False)

        if config.fast_dim is not None and config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        else:
            self.fast_project_in = nn.Identity()

        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = nn.ModuleList(FastTransformerBlock(config, use_sdpa=False) for _ in range(config.n_fast_layer))
        self.fast_norm = RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)

        self.register_buffer("fast_freqs_cis", precompute_freqs_cis(config.num_codebooks, config.fast_head_dim, config.rope_base), persistent=False)

        self.max_batch_size = -1
        self.max_seq_len = -1

    def setup_caches(self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16):
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_len, self.config.n_local_heads, self.config.head_dim, dtype=dtype)

        for b in self.fast_layers:
            b.attention.kv_cache = KVCache(max_batch_size, self.config.num_codebooks, self.config.fast_n_local_heads, self.config.fast_head_dim, dtype=dtype)

    def forward_generate(self, inp: Tensor, input_pos: Optional[Tensor] = None, **kwargs) -> BaseTransformerForwardResult:
        embeds = []
        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(inp[:, i + 1] + i * self.config.codebook_size)
            embeds.append(emb)

        vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)
        vq_masks = (inp[:, 0] >= self.tokenizer.semantic_begin_id) & (inp[:, 0] <= self.tokenizer.semantic_end_id)
        vq_embeds_sum[~vq_masks] = 0
        x = self.embeddings(inp[:, 0]) + vq_embeds_sum

        if input_pos is None:
            input_pos = torch.arange(inp.shape[-1], device=x.device)
            max_seq_len = inp.shape[-1]
        else:
            max_seq_len = self.max_seq_len

        mask = self.causal_mask[None, None, input_pos, :max_seq_len]
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        if x.size(1) > 1:
            x = x[:, -1:]

        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        x = self.fast_project_in(x)
        return BaseTransformerForwardResult(logits=token_logits, hidden_states=x)

    def forward_generate_fast(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        x = x.view(x.shape[0], 1, -1)
        fast_mask = self.causal_mask[None, None, input_pos, : self.config.num_codebooks]
        fast_freqs_cis = self.fast_freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis, fast_mask, input_pos=input_pos)

        fast_out = self.fast_norm(x)
        return self.fast_output(fast_out)

    @staticmethod
    def from_pretrained(path: str, load_weights: bool = False):
        config = DualARModelArgs.from_pretrained(str(path))
        tokenizer = FishTokenizer.from_pretrained(path)
        log_info(f"Loading model from {path}, config: {config}")
        model = DualARTransformer(config, tokenizer=tokenizer)

        if load_weights:
            weights = torch.load(Path(path) / "model.pth", map_location="cpu", mmap=True, weights_only=True)
            if "state_dict" in weights:
                weights = weights["state_dict"]
            if next(iter(weights.keys())).startswith("model."):
                weights = OrderedDict((k.replace("model.", ""), v) for k, v in weights.items())
            for k in list(weights.keys()):
                if "audio_" in k:
                    weights.pop(k)
            err = model.load_state_dict(weights, strict=False, assign=True)
            log_info(f"Model weights loaded - Status: {err}")

        return model


# ============= Generation =============

def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature, top_p, repetition_penalty, previous_tokens=None):
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)
    return torch.nn.functional.softmax(logits, dim=-1)


def sample(logits, temperature, top_p, repetition_penalty, previous_tokens=None):
    probs = logits_to_probs(logits=logits[0, -1], temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, previous_tokens=previous_tokens)
    return multinomial_sample_one_no_sync(probs), probs


def decode_one_token_ar(model, x, input_pos, temperature, top_p, repetition_penalty, audio_masks, audio_parts, previous_tokens=None):
    forward_result = model.forward_generate(x, input_pos, audio_masks=audio_masks, audio_parts=audio_parts)
    logits = forward_result.logits
    hidden_states = forward_result.hidden_states

    codebooks = [sample(logits, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, previous_tokens=(previous_tokens[:, 0] if previous_tokens is not None else None))[0]]

    for layer in model.fast_layers:
        if hasattr(layer, "attention") and hasattr(layer.attention, "kv_cache"):
            layer.attention.kv_cache.k_cache.fill_(0)
            layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor([codebook_idx], device=hidden_states.device, dtype=torch.long)
        logits = model.forward_generate_fast(hidden_states, input_pos)
        short_logits = logits[:, :, :1024]
        a = sample(short_logits, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, previous_tokens=(previous_tokens[codebook_idx + 1] if previous_tokens is not None else None))[0]
        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    return torch.stack(codebooks, dim=1).T


def estimate_output_tokens(text: str, tokens_per_char: float = 4.0) -> int:
    """
    Estimate the number of output tokens based on input text length.

    Based on empirical observation:
    - Chinese text: ~4 tokens per character
    - The actual ratio may vary based on speaking speed and content

    Args:
        text: Input text to synthesize
        tokens_per_char: Estimated tokens per character (default 4.0)

    Returns:
        Estimated number of tokens
    """
    # Count characters (excluding spaces and punctuation for better estimation)
    char_count = len([c for c in text if c.strip() and not c in '，。！？、；：""''【】（）《》…—'])
    # Add back some for punctuation pauses
    punct_count = len([c for c in text if c in '，。！？；：…'])

    estimated = int((char_count + punct_count * 0.5) * tokens_per_char)
    return max(estimated, 50)  # Minimum 50 tokens


def decode_n_tokens(model, cur_token, input_pos, num_new_tokens, temperature, top_p, repetition_penalty, audio_masks, audio_parts, decode_one_token=decode_one_token_ar, estimated_tokens=None):
    previous_tokens = torch.zeros((model.config.num_codebooks + 1, model.config.max_seq_len), dtype=torch.int, device=cur_token.device)

    # Use estimated tokens for progress bar if available
    total = estimated_tokens if estimated_tokens else None
    pbar = tqdm(total=total, desc="Generating", unit=" tokens")

    for i in range(num_new_tokens):
        win_size = 16
        window = previous_tokens[:, :win_size] if i < win_size else previous_tokens[:, i - win_size : i]

        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(model=model, x=cur_token, input_pos=input_pos, previous_tokens=window, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, audio_masks=audio_masks, audio_parts=audio_parts).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(model.config.num_codebooks + 1, -1)
        pbar.update(1)

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break

    # Update total to actual count for final display
    if pbar.total is None or pbar.total != i + 1:
        pbar.total = i + 1
        pbar.refresh()
    pbar.close()

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(*, model, prompt, max_new_tokens, audio_masks, audio_parts, decode_one_token=decode_one_token_ar, num_samples=1, estimated_tokens=None, **sampling_kwargs):
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}")

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
    else:
        max_new_tokens = model.config.max_seq_len - T

    device, dtype = prompt.device, prompt.dtype

    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len, dtype=next(model.parameters()).dtype)
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty((codebook_dim, model.config.max_seq_len), dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty

    temperature = getattr(model, "fixed_temperature", torch.tensor(0.8, device=device, dtype=torch.float))
    top_p = getattr(model, "fixed_top_p", torch.tensor(0.8, device=device, dtype=torch.float))
    repetition_penalty = getattr(model, "fixed_repetition_penalty", torch.tensor(1.1, device=device, dtype=torch.float))

    temp_val = sampling_kwargs.get("temperature", 0.7)
    top_p_val = sampling_kwargs.get("top_p", 0.7)
    rep_val = sampling_kwargs.get("repetition_penalty", 1.5)

    if abs(temperature.item() - temp_val) > 1e-6:
        temperature.fill_(temp_val)
    if abs(top_p.item() - top_p_val) > 1e-6:
        top_p.fill_(top_p_val)
    if abs(repetition_penalty.item() - rep_val) > 1e-6:
        repetition_penalty.fill_(rep_val)

    first_token = decode_one_token_ar(model, prompt.view(1, codebook_dim, -1), input_pos, temperature, top_p, repetition_penalty, audio_masks, audio_parts)
    seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    x = decode_n_tokens(model, first_token.view(1, codebook_dim, -1), input_pos, max_new_tokens - 1, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, audio_masks=audio_masks, audio_parts=audio_parts, decode_one_token=decode_one_token, estimated_tokens=estimated_tokens)
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    return seq


def init_model(checkpoint_path, device, precision, compile=False):
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)
    model = model.to(device=device, dtype=precision)
    log_info(f"Restored model from checkpoint")

    decode_one_token = decode_one_token_ar

    model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)
    model._cache_setup_done = False

    if compile and platform.system() == "Windows":
        log_warning("Compile mode is not supported on Windows. Disabling compilation.")
        compile = False

    if compile:
        log_info("Compiling function...")
        decode_one_token = torch.compile(decode_one_token, backend="inductor" if torch.cuda.is_available() else "aot_eager", mode="reduce-overhead" if torch.cuda.is_available() else None, fullgraph=True)

    return model.eval(), decode_one_token


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def generate_long(*, model, device, decode_one_token, text, num_samples=1, max_new_tokens=0, top_p=0.8, repetition_penalty=1.1, temperature=0.8, compile=False, prompt_text=None, prompt_tokens=None, **kwargs):
    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    if use_prompt:
        assert len(prompt_text) == len(prompt_tokens)

    if prompt_tokens:
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    tokenizer = model.tokenizer
    base_content_sequence = ContentSequence(modality="interleave")

    max_length = model.config.max_seq_len
    if use_prompt:
        for t, c in zip(prompt_text, prompt_tokens):
            base_content_sequence.append([TextPart(text=t), VQPart(codes=c)], add_end=True, speaker=0)
    base_content_sequence.append([TextPart(text=text)], add_end=False, speaker=0)

    encoded, audio_masks, audio_parts = base_content_sequence.encode_for_inference(tokenizer, num_codebooks=model.config.num_codebooks)
    if encoded.size(1) > max_length - 2048:
        raise ValueError(f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}")

    encoded = encoded.to(device=device)
    log_info(f"Encoded text: {text}")

    # Estimate output tokens for progress bar
    estimated_tokens = estimate_output_tokens(text)

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        prompt_length = encoded.size(1)
        t0 = time.perf_counter()

        y = generate(model=model, prompt=encoded, max_new_tokens=max_new_tokens, audio_masks=audio_masks, audio_parts=audio_parts, decode_one_token=decode_one_token, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, estimated_tokens=estimated_tokens)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t = time.perf_counter() - t0
        tokens_generated = y.size(1) - prompt_length
        log_info(f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_generated / t:.02f} tokens/sec")

        if torch.cuda.is_available():
            log_info(f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

        codes = y[1:, prompt_length:-1].clone()
        yield GenerateResponse(action="sample", codes=codes, text=text)

        del y, codes
        yield GenerateResponse(action="next")


# Note: Use tts.py as the main entry point for TTS inference
