# ruff: noqa: D100, D101, D102
from __future__ import annotations

from einops import rearrange
from hunterFormsBS.attend import Attend, FeedForward
from hunterFormsBS.theTypes import ParametersAttention
from hunterMakesPy import raiseIfNone
from hyper_connections import get_init_and_expand_reduce_stream_functions  # NOTE There is a newer version.
from more_itertools import loops
from operator import neg
from PoPE_pytorch import flash_attn_with_pope, PoPE
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import default, exists
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
	from rotary_embedding_torch import RotaryEmbedding

class Attention(nn.Module):
	def __init__(
		self,
		dim: int,
		dim_head: int = 64,
		attn_dropout: float = 0.0,
		heads: int = 8,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
		scale: float | None = None,
		*,
		flash: bool = True,
		sage_attention: bool = False,
		use_value_residual_learning: bool = False,
		learned_value_residual_mix: bool | None = None,
	) -> None:
		super().__init__()

		self.heads: int = heads
		self.norm: RMSNorm = RMSNorm(dim)
		self.pope_embed: PoPE | None = pope_embed
		self.rotary_embed: RotaryEmbedding | None = rotary_embed
		self.scale: float = scale or dim_head ** neg(0.5)

		self.attend: Attend = Attend(attn_dropout=attn_dropout, scale=self.scale, flash=flash, sage_attention=sage_attention)
		self.to_gates: nn.Linear = nn.Linear(dim, self.heads)

		dim_inner: int = self.heads * dim_head

		self.to_qkv: nn.Linear = nn.Linear(in_features=dim, out_features=dim_inner * 3, bias=False)
		self.to_out: nn.Sequential = nn.Sequential(nn.Linear(in_features=dim_inner, out_features=dim, bias=False), nn.Dropout(attn_dropout))

		self.learned_value_residual_mix: nn.Linear | None = None

		if ((use_value_residual_learning is True and learned_value_residual_mix is None)
			or (learned_value_residual_mix is True)):
			self.learned_value_residual_mix = nn.Linear(dim, self.heads)

		"""Original code because I have not tested the new code:
		if exists(learned_value_residual_mix):
			use_value_residual_learning = learned_value_residual_mix

		self.learned_value_residual_mix = nn.Linear(dim, self.heads) if use_value_residual_learning else None"""

	def forward(self, x: Tensor, value_residual: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
		x = self.norm(x)

		q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

		original_values: Tensor = v

		if exists(self.learned_value_residual_mix):
			mix: Tensor = self.learned_value_residual_mix(x)
			mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()
			v: Tensor = v.lerp(raiseIfNone(value_residual), mix)

		if exists(self.pope_embed):
			out: Tensor = flash_attn_with_pope(q, k, v, pos_emb=self.pope_embed(q.shape[-2]), softmax_scale=self.scale)
		elif exists(self.rotary_embed):
			q: Tensor = self.rotary_embed.rotate_queries_or_keys(q)
			k: Tensor = self.rotary_embed.rotate_queries_or_keys(k)
			out = self.attend(q, k, v)
		else:
			out = self.attend(q, k, v)

		# after attend
		gates: Tensor = self.to_gates(x)
		out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

		out = rearrange(out, 'b h n d -> b n (h d)')
		return self.to_out(out), original_values

class Transformer(Module):
	def __init__(
		self,
		*,
		use_value_residual_learning: bool = False,
		attn_dropout: float = 0.0,
		depth: int,
		dim_head: int = 64,
		dim: int,
		ff_dropout: float = 0.0,
		ff_mult: float | None = 4,
		flash_attn: bool = True,
		heads: int = 8,
		linear_attn: bool = False,  # noqa: ARG002
		mc_hyper_conn_sinkhorn_iters: int | None = None,  # noqa: ARG002
		norm_output: bool = True,
		num_residual_fracs: int | None = None,  # noqa: ARG002
		num_residual_streams: int = 1,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
		sage_attention: bool = False,
		scale: float | None = None,
	) -> None:
		super().__init__()

		parametersAttention = ParametersAttention(
			dim_head=dim_head,
			dim=dim,
			attn_dropout=attn_dropout,
			flash=flash_attn,
			heads=heads,
			pope_embed=pope_embed,
			rotary_embed=rotary_embed,
			sage_attention=sage_attention,
			scale=scale,
		)

		self.layers = ModuleList(
			ModuleList([Attention(**parametersAttention, use_value_residual_learning=use_value_residual_learning)
					, FeedForward(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout)])
				for _deep in loops(depth)
		)

		if num_residual_streams != 1:
			init_hyper_conn, *_streams = get_init_and_expand_reduce_stream_functions(
				num_residual_streams, disable=(num_residual_streams == 1)
			)
			for moduleList in self.layers:
				moduleList = cast('ModuleList', moduleList)
				for nnModule in moduleList:
					init_hyper_conn(dim=dim, branch=nnModule)

		"""Original code because I have not tested the new code:
		self.layers = ModuleList([])
		for _deep in loops(depth):
			attn = Attention(**parametersAttention, use_value_residual_learning=use_value_residual_learning)
			if num_residual_streams != 1:
				attn = init_hyper_conn(dim=dim, branch=attn)

			ff = FeedForward(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout)
			if num_residual_streams != 1:
				ff = init_hyper_conn(dim=dim, branch=ff)

			self.layers.append(ModuleList([attn, ff]))"""

		self.norm: RMSNorm | nn.Identity = RMSNorm(dim) if norm_output else nn.Identity()

	def forward(self, x: Tensor, value_residual: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
		first_values: Tensor | None = None
		if value_residual is not None:
			for sherpa in self.layers:
				attn: Attention = cast('Attention', cast('ModuleList', sherpa)[0])
				ff: FeedForward = cast('FeedForward', cast('ModuleList', sherpa)[1])
				x, next_values = attn(x, value_residual=value_residual)
				first_values = default(first_values, next_values)
				x = ff(x)
		else:
			# Compatibility with old weights
			for sherpa in self.layers:
				attn: Attention = cast('Attention', cast('ModuleList', sherpa)[0])
				ff: FeedForward = cast('FeedForward', cast('ModuleList', sherpa)[1])
				attn_out, next_values = attn(x, value_residual=value_residual)
				first_values = default(first_values, next_values)
				x = attn_out + x
				x = ff(x) + x

		return self.norm(x), first_values
