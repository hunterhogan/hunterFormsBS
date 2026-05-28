"""Apply feedforward and transformer residual blocks for music source separation.

You can use this module to build the shared position-wise feedforward block and the stacked
attention-and-feedforward transformer used by
`hunterFormsBS.bandSplitRotator.BandSplitRotator` [1]. `FeedForward` applies a configurable
width-expansion nonlinear transform to every sequence position independently. `Transformer`
stacks repeated `Attention` [2] and `FeedForward` pairs into a residual sequence and
optionally normalizes the result.

Contents
--------
Classes
	FeedForward
		Apply a position-wise expansion-and-projection nonlinear block.
	Transformer
		Stack repeated `Attention` and `FeedForward` pairs into a residual sequence.

References
----------
[1] `hunterFormsBS.bandSplitRotator.BandSplitRotator`

[2] `hunterFormsBS.attend.Attention`
"""
from __future__ import annotations

from hunterFormsBS.attend import Attention
from hunterFormsBS.theTypes import ParametersAttention
from more_itertools import loops
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
	from PoPE_pytorch import PoPE
	from rotary_embedding_torch import RotaryEmbedding

class FeedForward(Module):
	"""Transform activations with a position-wise expansion-and-projection block.

	You can use `FeedForward` after `Attention` [1] inside `Transformer` [2] when activation `Tensor`
	`x` at each sequence position should pass through the same nonlinear feature transform.
	`FeedForward` preserves the batch axis, sequence axis, and feature axis of activation `Tensor` `x`
	while expanding the feature width internally before projecting back to the original width. The
	shared `BandSplitRotator` stack uses one `FeedForward` block after every attention block [3].

	Attributes
	----------
	net : nn.Sequential
		Position-wise submodule sequence containing root-mean-square normalization, width expansion,
		`nn.GELU`, `ff_dropout`, width projection, and output `ff_dropout` [4].

	References
	----------
	[1] `Attention`

	[2] `Transformer`

	[3] `hunterFormsBS.bandSplitRotator.BandSplitRotator`

	[4] PyTorch.
		https://context7.com/pytorch/pytorch
	"""
	def __init__(self, dim: int, ff_mult: float | None = 4.0, ff_dropout: float = 0.0) -> None:
		"""Set up a position-wise feedforward block for feature width `dim`.

		You can use `__init__` to choose the input and output feature width `dim`, the internal width
		expansion factor `ff_mult`, and the `ff_dropout` probability for the shared feedforward
		sublayer inside `Transformer` [1]. `__init__` stores the resulting normalization, linear,
		activation, and `ff_dropout` layers in `net` for reuse by later calls to `forward`.

		Parameters
		----------
		dim : int
			Input feature width before expansion and output feature width after projection.
		ff_mult : float | None = 4.0
			Multiplicative factor used to compute the hidden feature width `int(dim * ff_mult)`. When
			`ff_mult` is `None`, `__init__` uses `4.0`.
		ff_dropout : float = 0.0
			Dropout probability applied after the hidden activation and after the output projection.

		PyTorch
		-------
		submodule construction : implementation detail
			`__init__` computes `dim_inner = int(dim * ff_mult)` and stores `net` as a six-stage
			`nn.Sequential` with root-mean-square normalization, `nn.Linear(dim, dim_inner)`,
			`nn.GELU`, `nn.Dropout(ff_dropout)`, `nn.Linear(dim_inner, dim)`, and a second
			`nn.Dropout(ff_dropout)` [2].

		References
		----------
		[1] `Transformer`

		[2] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		super().__init__()
		if ff_mult is None:
			ff_mult = 4.0
		dim_inner: int = int(dim * ff_mult)
		self.net: nn.Sequential = nn.Sequential(
			RMSNorm(dim), nn.Linear(dim, dim_inner), nn.GELU(), nn.Dropout(ff_dropout), nn.Linear(dim_inner, dim), nn.Dropout(ff_dropout)
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Transform activations `x` with the position-wise feedforward block.

		You can use `forward` when `x` already has batch axis, sequence axis, and feature axis.
		`forward` applies the same nonlinear feature transform to every position independently, reuses
		the submodule sequence configured by `__init__`, and returns an activation `Tensor` with the
		same outer shape as `x`. `forward` supplies the position-wise branch used by `Transformer`
		[1].

		Parameters
		----------
		x : Tensor
			Input activation `Tensor` with shape `batch × sequence position × feature`.

		Returns
		-------
		out : Tensor
			Output activation `Tensor` with shape `batch × sequence position × feature`.

		Shape Transformation
		--------------------
		position-wise mapping : implementation detail
			`forward` preserves the outer shape `B×N×D`. `forward` applies the same feedforward map to
			each position vector `x[b, n, :]` independently of every other position.

		References
		----------
		[1] `Transformer`
		"""
		return self.net(x)

class Transformer(Module):
	"""Refine activations with repeated attention-and-feedforward blocks.

	You can use `Transformer` as the shared sequence stack inside `BandSplitRotator` [1].
	`Transformer` takes activation `Tensor` `x`, repeats `depth` pairs of `Attention` [2] and
	`FeedForward` [3], and optionally normalizes the result. `Transformer`
	preserves the batch axis, sequence axis, and feature axis of activation `Tensor` `x`. The
	constructor still accepts several compatibility parameters, but the stack itself no longer
	contains `LinearAttention`.

	Attributes
	----------
	layers : ModuleList
		Sequence of per-depth `[attention, feedforward]` pairs, where the attention entry is
		`Attention` [2] and the feedforward entry is `FeedForward` [3].
	norm : RMSNorm | nn.Identity
		Final output normalization module applied after the last residual block when `norm_output` is
		`True`, or `nn.Identity` otherwise.

	References
	----------
	[1] `hunterFormsBS.bandSplitRotator.BandSplitRotator`

	[2] `Attention`

	[3] `FeedForward`
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
		linear_attn: bool = False,  # noqa: ARG002
		norm_output: bool = True,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
		sage_attention: bool = False,
		scale: float | None = None,
	) -> None:
		"""Set up a transformer stack for feature width `dim` and layer count `depth`.

		You can use `__init__` to choose the stack depth, model width, attention-head layout, dropout
		probabilities, attention backend, and position-encoding path for a
		shared transformer block. `__init__` stores `depth` pairs of `Attention` [1] and `FeedForward`
		[2] in `layers`, forwards the downstream attention parameters without renaming through
		`ParametersAttention` [3], and configures optional output normalization in `norm`. `__init__`
		retains `linear_attn` only for configuration compatibility with earlier versions.

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
			Dropout probability passed to each `FeedForward` block [2].
		ff_mult : float | None = 4
			Hidden-width expansion factor passed to each `FeedForward` block [2].
		flash_attn : bool = True
			Whether each attention sublayer may use PyTorch SDPA (scaled dot-product attention)
			backends [4].
		heads : int = 8
			Number of attention heads in each attention sublayer.
		linear_attn : bool = False
			Vestigial flag retained for configuration compatibility passthrough. This unified implementation
			ignores `linear_attn` because `LinearAttention` was removed.
		norm_output : bool = True
			Whether to apply final output normalization after the last residual block.
		pope_embed : PoPE | None = None
			Optional polar-coordinate position encoder passed to each `Attention` [1] [5][6]. When
			both positional encoders are present, `Attention.forward` prefers `pope_embed`.
		rotary_embed : RotaryEmbedding | None = None
			Optional rotary positional encoder passed to each `Attention` [1] [7].
		sage_attention : bool = False
			Whether each `Attention` should prefer SageAttention [8][9]. `hunterFormsBS` does not
			install `sageattention`.
		scale : float | None = None
			Optional attention-score multiplier override passed to each `Attention` [1].

		PyTorch
		-------
		layer construction : implementation detail
			`__init__` first builds `ParametersAttention` [3] with the downstream attention arguments.
			`__init__` then creates `depth` `ModuleList` pairs, each containing one
			`Attention(**parametersAttention)` [1] and one `FeedForward(dim=dim, ff_mult=ff_mult,
			ff_dropout=ff_dropout)` [2]. `norm` becomes `RMSNorm(dim)` when
			`norm_output` is `True` and `nn.Identity()` otherwise [4].

		References
		----------
		[1] `Attention`

		[2] `FeedForward`

		[3] `hunterFormsBS.theTypes.ParametersAttention`

		[4] PyTorch.
			https://context7.com/pytorch/pytorch
		[5] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
			Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings.
			https://arxiv.org/abs/2509.10534
		[6] lucidrains/PoPE-pytorch.
			https://github.com/lucidrains/PoPE-pytorch
		[7] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
			RoFormer: Enhanced Transformer with Rotary Position Embedding.
			https://arxiv.org/abs/2104.09864
		[8] Zhang, J., Wei, J., Huang, H., Zhang, P., Zhu, J., and Chen, J. (2025).
			SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.
			https://arxiv.org/abs/2410.02367
		[9] thu-ml/SageAttention
			https://github.com/thu-ml/SageAttention
		"""
		super().__init__()

		parametersAttention = ParametersAttention(
			attn_dropout=attn_dropout,
			dim_head=dim_head,
			dim=dim,
			flash=flash_attn,
			heads=heads,
			pope_embed=pope_embed,
			rotary_embed=rotary_embed,
			sage_attention=sage_attention,
			scale=scale,
		)

		self.layers = ModuleList(
			ModuleList([Attention(**parametersAttention)
					, FeedForward(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout)])
				for _layer in loops(depth)
		)

		self.norm: RMSNorm | nn.Identity = RMSNorm(dim) if norm_output else nn.Identity()

	def forward(self, x: Tensor) -> Tensor:
		"""Transform activations `x` through the residual stack.

		You can use `forward` when `x` already has batch axis, sequence axis, and feature axis.

		Parameters
		----------
		x : Tensor
			Input activation `Tensor` with shape `batch × sequence position × feature`.

		Returns
		-------
		result : Tensor
			Output activation `Tensor` with shape `batch × sequence position × feature`.

		References
		----------
		[1] `Attention.forward`

		[2] `FeedForward.forward`
		"""
		for sherpa in self.layers:
			attn: Attention = cast('Attention', cast('ModuleList', sherpa)[0])
			ff: FeedForward = cast('FeedForward', cast('ModuleList', sherpa)[1])
			x = attn(x) + x
			x = ff(x) + x

		return self.norm(x)
