"""Provide attention, feedforward, and transformer blocks for music source separation.

You can use this module to assemble the hierarchical attention stack shared by BS-RoFormer [1] and
Mel-Band RoFormer [2]. The module provides four building-block classes. `Attend` evaluates attention
from precomputed query, key, and value arrays and optionally delegates to PyTorch SDPA backends [3].
`Attention` projects activations into query, key, and value arrays, applies RoPE [4] or PoPE [5]
positional encoding, gates head outputs, and calls `Attend`. `FeedForward` applies a position-wise
expansion-and-projection block after each attention sublayer. `LinearAttention` provides the optional
XCiT-style cross-covariance attention pre-block [6]. `Transformer` stacks those blocks into the
repeated attention-and-feedforward sequence consumed by `BandSplitRotator` [7].

Contents
--------
Classes
	Attend
		Evaluate attention output from precomputed query, key, and value arrays using an explicit or
		PyTorch SDPA path.
	Attention
		Project activations into multi-head query, key, and value arrays, apply positional encoding,
		and return gated attention output.
	FeedForward
		Apply a position-wise expansion-and-projection nonlinear block.
	LinearAttention
		Apply XCiT-style cross-covariance attention across feature channels.
	Transformer
		Stack attention and feedforward sublayers into a repeated residual sequence.

References
----------
[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
	Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
[2] Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation.
	https://arxiv.org/abs/2409.04702
[3] PyTorch.
	https://context7.com/pytorch/pytorch
[4] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
	Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
[5] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., & Mozer, M. C. (2025). Decoupling the
	"What" and "Where" With Polar Coordinate Positional Embeddings. https://arxiv.org/abs/2509.10534
[6] El-Nouby, A., Touvron, H., Caron, M., Bojanowski, P., Douze, M., Joulin, A., Laptev, I.,
	Neverova, N., Synnaeve, G., Verbeek, J., & Jégou, H. (2021). XCiT: Cross-Covariance Image
	Transformers. https://arxiv.org/abs/2106.09681
[7] hunterFormsBS.bandSplitRotator.BandSplitRotator
"""
from __future__ import annotations

from hunterFormsBS.attend import Attention, FeedForward
from hunterFormsBS.theTypes import ParametersAttention
from hyper_connections import get_init_and_expand_reduce_stream_functions  # NOTE There is a newer version.
from more_itertools import loops
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import default
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
	from PoPE_pytorch import PoPE
	from rotary_embedding_torch import RotaryEmbedding

class Transformer(Module):
	"""Refine activations with a repeated attention-and-feedforward stack.

	You can use `Transformer` as the shared sequence stack inside BS-RoFormer [4] and
	Mel-Band RoFormer [5]. `Transformer` takes activation `Tensor` `x`, repeats `depth` pairs of
	attention and feedforward sublayers, uses `Attention` [1] by default or `LinearAttention` [2]
	when `linear_attn=True`, and optionally normalizes the result. `Transformer` preserves the
	batch axis, sequence axis, and feature axis of activation `Tensor` `x`.

	Attributes
	----------
	layers : ModuleList
		Sequence of per-depth `[attention, feedforward]` pairs, where the attention entry is
		`Attention` [1] or `LinearAttention` [2] and the feedforward entry is `FeedForward` [3].
	norm : RMSNorm | nn.Identity
		Final output normalization module applied after the last residual block when `norm_output` is
		`True`, or `nn.Identity` otherwise.

	See Also
	--------
	Attention
		Provide the default sequence-position attention branch used by `Transformer`.
	LinearAttention
		Provide the optional cross-covariance attention branch used when `linear_attn=True`.
	FeedForward
		Provide the position-wise feature transform paired with each attention layer.

	References
	----------
	[1] hunterFormsBS.attend.Attention

	[2] hunterFormsBS.attend.LinearAttention

	[3] hunterFormsBS.attend.FeedForward

	[4] Lu, W.-T., Wang, J.-C., Kong, Q., and Hung, Y.-N. (2023).
		Music Source Separation with Band-Split RoPE Transformer.
		https://doi.org/10.48550/arXiv.2309.02612
	[5] Wang, J.-C., Lu, W.-T., and Won, M. (2023).
		Mel-Band RoFormer for Music Source Separation.
		https://arxiv.org/abs/2409.04702
	"""
	def __init__(
		self,
		*,
		depth: int,
		attn_dropout: float = 0.0,
		dim_head: int = 64,
		dim: int,
		ff_dropout: float = 0.0,
		ff_mult: float | None = 4,
		flash_attn: bool = True,
		heads: int = 8,
		learned_value_residual_mix: bool | None = None,
		linear_attn: bool = False,  # noqa: ARG002
		mc_hyper_conn_sinkhorn_iters: int | None = None,  # noqa: ARG002
		norm_output: bool = True,
		num_residual_fracs: int | None = None,  # noqa: ARG002
		num_residual_streams: int = 1,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
		sage_attention: bool = False,
		scale: float | None = None,
		use_value_residual_learning: bool = False,
	) -> None:
		"""Set up a transformer stack for feature width `dim` and layer count `depth`.

		You can use `__init__` to choose the stack depth, model width, attention-head layout, dropout
		probabilities, attention backend, and positional-encoding path for a shared transformer
		block. `__init__` stores `depth` pairs of attention and feedforward sublayers in `layers` and
		configures optional output normalization in `norm`.

		Parameters
		----------
		attn_dropout : float = 0.0
			Dropout probability passed to each attention sublayer.
		depth : int
			Number of repeated attention-plus-feedforward residual blocks.
		dim_head : int = 64
			Feature width of each attention head.
		dim : int
			Input and output feature width of the stack.
		ff_dropout : float = 0.0
			Dropout probability passed to each `FeedForward` block [3].
		ff_mult : float = 4
			Hidden-width expansion factor passed to each `FeedForward` block [3].
		flash_attn : bool = True
			Whether each attention sublayer may use PyTorch SDPA (scaled dot-product attention)
			backends [4].
		heads : int = 8
			Number of attention heads in each attention sublayer.
		linear_attn : bool = False
			Whether to build each attention sublayer with `LinearAttention` [2] instead of
			`Attention` [1].
		norm_output : bool = True
			Whether to apply final output normalization after the last residual block.
		pope_embed : PoPE | None = None
			Optional polar coordinate position encoder passed to `Attention` [1] when
			`linear_attn=False`. `pope_embed` is ignored when `linear_attn=True`.
		rotary_embed : RotaryEmbedding | None = None
			Optional rotary positional encoder passed to `Attention` [1] when `linear_attn=False`.
			`rotary_embed` is ignored when `linear_attn=True`.

		PyTorch
		-------
		layer construction : implementation detail
			`__init__` builds `depth` `ModuleList` pairs. Each pair stores one attention sublayer and
			one feedforward sublayer. When `linear_attn=True`, the attention sublayer is
			`LinearAttention(**attentionKwargs)` [2]. Otherwise, `__init__` creates `Attention` [1]
			and passes `rotary_embed` plus `pope_embed` to each `Attention`. The paired feedforward
			sublayer is `FeedForward(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout)` [3]. `norm` becomes
			`RMSNorm(dim)` when `norm_output` is `True` and `nn.Identity()` otherwise [4].

		References
		----------
		[1] hunterFormsBS.attend.Attention

		[2] hunterFormsBS.attend.LinearAttention

		[3] hunterFormsBS.attend.FeedForward

		[4] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		super().__init__()

		parametersAttention = ParametersAttention(
			attn_dropout=attn_dropout,
			dim_head=dim_head,
			dim=dim,
			flash=flash_attn,
			heads=heads,
			learned_value_residual_mix=learned_value_residual_mix,
			pope_embed=pope_embed,
			rotary_embed=rotary_embed,
			sage_attention=sage_attention,
			scale=scale,
			use_value_residual_learning=use_value_residual_learning,
		)

		self.layers = ModuleList(
			ModuleList([Attention(**parametersAttention)
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
		"""Transform activations `x` through the residual stack.

		You can use `forward` when `x` already has batch axis, sequence axis, and feature axis.
		`forward` applies each stored `Attention` [1] or `LinearAttention` [2] sublayer and each
		stored `FeedForward` [3] sublayer with residual additions and then applies `norm`. `forward`
		returns an activation `Tensor` with the same outer shape as `x`.

		Parameters
		----------
		x : Tensor
			Input activation `Tensor` with shape `batch × sequence position × feature`.

		Returns
		-------
		transformed_x : Tensor
			Output activation `Tensor` with shape `batch × sequence position × feature`.

		Mathematics
		-----------
		residual recurrence : equation
		```
			Let L ≜ `len(self.layers)`,  X₀ ≜ `x`,
				Aₗ ≜ `self.layers[ℓ][0]`,  Fₗ ≜ `self.layers[ℓ][1]`,  ℓ ∈ {0, …, L−1},
				N ≜ `self.norm`

			Hₗ = Aₗ(Xₗ) + Xₗ
			Xₗ₊₁ = Fₗ(Hₗ) + Hₗ
			Y = N(X_L)

			where  Y ≜ `transformed_x`
		```

		References
		----------
		[1] hunterFormsBS.attend.Attention

		[2] hunterFormsBS.attend.LinearAttention

		[3] hunterFormsBS.attend.FeedForward
		"""
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
