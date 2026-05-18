"""Define shared typed configuration records for `hunterFormsBS`.

You can use this module to access the small record classes that group related keyword arguments and
pass those keyword arguments through several constructor layers without renaming fields. The records
keep the attention stack, the STFT helpers, and the separator wrappers aligned when one outer model
configures several downstream classes. The records are consumed by `hunterFormsBS.attend` [1],
`hunterFormsBS.bandSplit` [2], `hunterFormsBS.bandSplitRotator.BandSplitRotator` [3],
`hunterFormsBS.bs_roformer.BSRoformer` [4], and `hunterFormsBS.mel_band_roformer.MelBandRoformer` [5].

Contents
--------
Classes
	FlashAttentionConfig
		Store one scaled-dot-product-attention backend selection record.
	ParametersAttention
		Collect one attention-block keyword record.
	ParametersComputeLoss
		Collect one multi-resolution STFT loss keyword record.
	ParametersSTFT
		Collect one forward-and-inverse STFT keyword record.
	ParametersTransformer
		Collect one transformer-block keyword record.

References
----------
[1] hunterFormsBS.attend

[2] hunterFormsBS.bandSplit

[3] hunterFormsBS.bandSplitRotator.BandSplitRotator

[4] hunterFormsBS.bs_roformer.BSRoformer

[5] hunterFormsBS.mel_band_roformer.MelBandRoformer
"""
from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable
	from PoPE_pytorch import PoPE
	from rotary_embedding_torch import RotaryEmbedding
	from torch import Tensor

class FlashAttentionConfig(NamedTuple):
	"""Store one scaled-dot-product-attention backend selection record.

	You can use `FlashAttentionConfig` to keep the backend-enable flags together when one attention
	block selects which scaled-dot-product-attention backend may run on CPU or CUDA.

	Attributes
	----------
	enable_flash : bool
		Whether the flash backend may run.
	enable_math : bool
		Whether the math backend may run.
	enable_mem_efficient : bool
		Whether the memory-efficient backend may run.
	"""
	enable_flash: bool
	enable_math: bool
	enable_mem_efficient: bool

class ParametersAttention(TypedDict):
	"""Collect one attention-block keyword record.

	You can use `ParametersAttention` to keep one attention-block configuration together while wrapper
	classes forward stable field names through several constructor layers. The record stores head
	geometry, dropout, positional-encoder objects, backend toggles, and optional
	value-residual-learning controls.

	Attributes
	----------
	attn_dropout : float
		Dropout probability applied inside the attention block.
	dim_head : int
		Feature width of each attention head before concatenation.
	dim : int
		Input and output feature width of the attention block.
	flash : bool
		Whether the attention block may use the optimized scaled-dot-product-attention backend path.
	heads : int
		Number of attention heads.
	learned_value_residual_mix : bool | None
		Compatibility switch controlling whether the attention block creates one learned mixer for
		value-residual learning.
	pope_embed : PoPE | None
		Optional polar-coordinate positional encoder object. `pope_embed` is mutually exclusive with
		`rotary_embed`.
	rotary_embed : RotaryEmbedding | None
		Optional rotary positional encoder object. `rotary_embed` is mutually exclusive with
		`pope_embed`.
	sage_attention : bool
		Whether the attention block may try the optional `SageAttention` backend [1]. `SageAttention`
		is not installed automatically with `hunterFormsBS`, so you must install `SageAttention`
		separately [1].
	scale : float | None
		Optional override for the default inverse-square-root attention-score scale.
	use_value_residual_learning : bool
		Whether the attention block enables value-residual learning.

	References
	----------
	[1] thu-ml/SageAttention
		https://github.com/thu-ml/SageAttention
	"""
	attn_dropout: float
	dim_head: int
	dim: int
	flash: bool
	heads: int
	learned_value_residual_mix: bool | None
	pope_embed: PoPE | None
	rotary_embed: RotaryEmbedding | None
	sage_attention: bool
	scale: float | None
	use_value_residual_learning: bool

class ParametersComputeLoss(TypedDict):
	"""Collect one multi-resolution STFT loss keyword record.

	You can use `ParametersComputeLoss` to keep the shared short-time Fourier transform (STFT) loss
	settings together. The record stores the hop size, each evaluation window size, the minimum FFT
	size, the normalization flag, the loss weight, and the window-construction callable.

	Attributes
	----------
	hop_length : int
		Hop size between adjacent STFT frames.
	loss_weight : float
		Multiplier for the multi-resolution STFT loss term.
	window_sizes : tuple[int, ...]
		Each STFT windowing-function size evaluated by the multi-resolution loss.
	n_fft : int
		Lower bound on the FFT size used for each STFT resolution.
	normalized : bool
		Whether each STFT uses normalized scaling.
	window_fn : Callable[..., Tensor]
		Callable that builds the STFT ***window***ing function for a requested size.
	"""
	hop_length: int
	loss_weight: float
	window_sizes: tuple[int, ...]
	n_fft: int
	normalized: bool
	window_fn: Callable[..., Tensor]

class ParametersSTFT(TypedDict):
	"""Collect one forward-and-inverse STFT keyword record.

	You can use `ParametersSTFT` to keep one short-time Fourier transform (STFT) configuration
	together. The same record can be forwarded to the forward STFT call and the inverse STFT call so
	both operations use the same FFT size, hop size, windowing-function length, and normalization
	rule.

	Attributes
	----------
	hop_length : int
		Hop size between adjacent STFT frames.
	n_fft : int
		FFT size used by the STFT.
	normalized : bool
		Whether the STFT uses normalized scaling.
	win_length : int
		Length of the STFT windowing function.
	"""
	hop_length: int
	n_fft: int
	normalized: bool
	win_length: int

class ParametersTransformer(TypedDict):
	"""Collect one transformer-block keyword record.

	You can use `ParametersTransformer` to keep one transformer-block configuration together while
	separator wrappers forward stable field names to the shared attention stack. The record stores
	attention geometry, feedforward settings, backend toggles, optional multi-stream residual
	settings, and compatibility fields kept for older configuration files.

	Attributes
	----------
	attn_dropout : float
		Dropout probability passed to each attention sublayer.
	dim : int
		Input and output feature width of the transformer block.
	dim_head : int
		Feature width of each attention head.
	ff_dropout : float
		Dropout probability in the feedforward sublayer.
	ff_mult : float | None
		Hidden-width expansion factor for each feedforward sublayer. `None` asks the feedforward
		sublayer to use its default expansion factor.
	flash_attn : bool
		Whether attention sublayers may use the optimized scaled-dot-product-attention backend path.
	heads : int
		Number of attention heads.
	learned_value_residual_mix : bool | None
		Compatibility switch controlling whether attention sublayers create one learned mixer for
		value-residual learning.
	linear_attn : bool
		Legacy compatibility flag preserved for older configuration files. The current shared
		transformer stack keeps only the standard attention path, so `linear_attn` is forwarded as
		metadata rather than selecting a separate linear-attention implementation.
	norm_output : bool
		Whether to normalize after the last layer.
	num_residual_streams : int
		Number of residual streams requested for multi-stream residual wrapping.
	sage_attention : bool
		Whether attention sublayers may try the optional `SageAttention` backend [1]. `SageAttention`
		is not installed automatically with `hunterFormsBS`, so you must install `SageAttention`
		separately [1].
	scale : float | None
		Optional override for the default inverse-square-root attention-score scale.

	References
	----------
	[1] thu-ml/SageAttention
		https://github.com/thu-ml/SageAttention
	"""
	attn_dropout: float
	dim_head: int
	dim: int
	ff_dropout: float
	ff_mult: float | None
	flash_attn: bool
	heads: int
	learned_value_residual_mix: bool | None
	linear_attn: bool
	norm_output: bool
	num_residual_streams: int
	sage_attention: bool
	scale: float | None
