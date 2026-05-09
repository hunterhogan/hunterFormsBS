# pyright: reportMissingParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: D100, ANN001, S101, ARG002 D101 D102
from __future__ import annotations

from einops import rearrange
from hunterFormsBS import KwargsOfAttention
from hunterFormsBS.attend import Attend, FeedForward, LinearAttention
from hyper_connections import get_init_and_expand_reduce_stream_functions  # NOTE There is a newer version.
from more_itertools import loops
from PoPE_pytorch import flash_attn_with_pope, PoPE
from rotary_embedding_torch import RotaryEmbedding
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import default, exists
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast

class Attention(nn.Module):
	def __init__(
		self,
		dim: int,
		heads: int = 8,
		dim_head: int = 64,
		dropout: float = 0.0,
		rotary_embed: RotaryEmbedding | None = None,
		pope_embed: PoPE | None = None,
		*,
		flash: bool = True,
		add_value_residual: bool = False,
		learned_value_residual_mix: bool | None = None,
	) -> None:
		super().__init__()
		self.heads: int = heads
		self.scale: float = dim_head**-0.5
		dim_inner: int = self.heads * dim_head

		self.rotary_embed: RotaryEmbedding | None = rotary_embed
		self.pope_embed: PoPE | None = pope_embed

		if exists(self.rotary_embed) and exists(self.pope_embed):
			message: str = "I received both `rotary_embed` and `pope_embed`, but these positional encodings are mutually exclusive."
			raise ValueError(message)

		self.attend: Attend = Attend(flash=flash, dropout=dropout)

		self.norm: RMSNorm = RMSNorm(dim)
		self.to_qkv: nn.Linear = nn.Linear(dim, dim_inner * 3, bias=False)

		if exists(learned_value_residual_mix):
			add_value_residual = learned_value_residual_mix

		self.learned_value_residual_mix = nn.Linear(dim, self.heads) if add_value_residual else None


		self.to_gates: nn.Linear = nn.Linear(dim, self.heads)
		self.to_out: nn.Sequential = nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout))

	def forward(self, x: Tensor, value_residual: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
		x = self.norm(x)

		q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

		original_values = v

		if exists(self.learned_value_residual_mix):
			mix = self.learned_value_residual_mix(x)
			mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()

			assert exists(value_residual)
			v = v.lerp(value_residual, mix)

		if exists(self.pope_embed):
			out: Tensor = flash_attn_with_pope(q, k, v, pos_emb=self.pope_embed(q.shape[-2]), softmax_scale=self.scale)
		elif exists(self.rotary_embed):
			q: Tensor = self.rotary_embed.rotate_queries_or_keys(q)
			k: Tensor = self.rotary_embed.rotate_queries_or_keys(k)
			out = self.attend(q, k, v)
		else:
			out = self.attend(q, k, v)

		gates: Tensor = self.to_gates(x)
		out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

		out = rearrange(out, 'b h n d -> b n (h d)')
		return self.to_out(out), original_values

class Transformer(Module):
	def __init__(
		self,
		*,
		add_value_residual=False,
		attn_dropout: float = 0.0,
		depth: int,
		dim_head: int = 64,
		dim: int,
		ff_dropout: float = 0.0,
		ff_mult: float = 4,
		flash_attn: bool = True,
		heads: int = 8,
		linear_attn: bool = False,
		mc_hyper_conn_sinkhorn_iters=None,
		norm_output: bool = True,
		num_residual_fracs=None,
		num_residual_streams=1,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
	) -> None:
		super().__init__()

		init_hyper_conn, *_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable=num_residual_streams == 1)

		attentionKwargs = KwargsOfAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn)

		self.layers = ModuleList([])
		for _deep in loops(depth):
			if linear_attn:
				attn = LinearAttention(**attentionKwargs)
			else:
				attn = Attention(**attentionKwargs, rotary_embed=rotary_embed, pope_embed=pope_embed, add_value_residual=add_value_residual)
				if num_residual_streams != 1:
					attn = init_hyper_conn(dim=dim, branch=attn)

			ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
			if num_residual_streams != 1:
				ff = init_hyper_conn(dim=dim, branch=ff)

			self.layers.append(ModuleList([attn, ff]))

		self.norm: RMSNorm | nn.Identity = RMSNorm(dim) if norm_output else nn.Identity()

	def forward(self, x: Tensor, value_residual: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
		first_values = None
		if value_residual is not None:
			for sherpa in self.layers:
				attn: Attention | LinearAttention = cast(Attention | LinearAttention, cast(ModuleList, sherpa)[0])
				ff: FeedForward = cast(FeedForward, cast(ModuleList, sherpa)[1])
				x, next_values = attn(x, value_residual=value_residual)
				first_values = default(first_values, next_values)
				x = ff(x)
		else:
			# Compatibility with old weights
			for sherpa in self.layers:
				attn: Attention | LinearAttention = cast(Attention | LinearAttention, cast(ModuleList, sherpa)[0])
				ff: FeedForward = cast(FeedForward, cast(ModuleList, sherpa)[1])
				attn_out, next_values = attn(x, value_residual=value_residual)
				first_values = default(first_values, next_values)
				x = attn_out + x
				x = ff(x) + x

		return self.norm(x), first_values
