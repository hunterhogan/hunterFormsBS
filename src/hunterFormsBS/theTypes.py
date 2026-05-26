"""Define shared typed configuration records for `hunterFormsBS`.

You can use this module to access the small record classes that group related keyword arguments and
pass those keyword arguments through several constructor layers without renaming fields. The records
keep the attention stack, the mask-estimation heads, the optional segmentation branch, the STFT
helpers, and the separator model aligned when one outer model configures several downstream classes.
The records are consumed by `hunterFormsBS.attend` [1], `hunterFormsBS.bandSplit` [2],
`hunterFormsBS.bandSplitRotator.BandSplitRotator` [3], and `hunterFormsBS.hyperACE` [4].

Contents
--------
Classes
	FlashAttentionConfig
		Store one scaled-dot-product-attention backend selection record.
	ParametersAttention
		Collect one attention-block keyword record.
	ParametersComputeLoss
		Collect one multi-resolution STFT loss keyword record.
	ParametersMaskEstimator
		Collect one mask-estimator keyword record.
	ParametersSTFT
		Collect one forward-and-inverse STFT keyword record.
	ParametersTransformer
		Collect one transformer-block keyword record.

References
----------
[1] `hunterFormsBS.attend`

[2] `hunterFormsBS.bandSplit`

[3] `hunterFormsBS.bandSplitRotator.BandSplitRotator`

[4] `hunterFormsBS.hyperACE`
"""
from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable
	from PoPE_pytorch import PoPE
	from rotary_embedding_torch import RotaryEmbedding
	from torch import nn, Tensor

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

	You can use `ParametersAttention` to keep one attention-block configuration together while outer
	model classes forward stable field names through several constructor layers. The record stores head
	geometry, dropout, positional-encoder objects, backend toggles, and attention-score scaling.

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
	pope_embed: PoPE | None
	rotary_embed: RotaryEmbedding | None
	sage_attention: bool
	scale: float | None

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

class ParametersMaskEstimator(TypedDict):
	"""Collect one mask-estimator keyword record.

	You can use `ParametersMaskEstimator` to keep the full `MaskEstimator` keyword family together
	while `BandSplitRotator` forwards stable field names into each mask-estimator head [1]. The
	record stores the shared band-local MLP settings, the `use_hyperACE` switch, and the `segm_*`
	parameter families for the optional segmentation-style branch [2].

	Attributes
	----------
	activation : type[nn.Module]
		Activation class instantiated inside each band-local `MLP` head.
	dim : int
		Feature width of each input band token.
	depth : int
		Number of hidden-width repeats inside each band-local `MLP` head.
	mlp_expansion_factor : int
		Hidden-width multiplier relative to `dim` for each band-local `MLP` head.
	use_hyperACE : bool
		Whether each mask-estimator head adds the optional segmentation-style branch [2].

	Other Parameters
	----------------
	segmentation output geometry : forwarded parameter family
		See `hunterFormsBS.hyperACE.SegmModel.__init__` [2]. `ParametersMaskEstimator` stores
		`segm_out_bins` and `segm_out_channels`.
	backbone parameters : forwarded parameter family
		See `hunterFormsBS.hyperACE.Backbone.__init__` [3]. `ParametersMaskEstimator` stores
		`segm_base_channels`, `segm_base_depth`, and `segm_backbone_channels`.
	hypergraph size and branch parameters : forwarded parameter family
		See `hunterFormsBS.hyperACE.HyperACE.__init__` [4]. `ParametersMaskEstimator` stores
		`segm_num_hyperedges`, `segm_num_heads`, and the `segm_hyperace_*` family.
	decoder parameters : forwarded parameter family
		See `hunterFormsBS.hyperACE.Decoder.__init__` [5]. `ParametersMaskEstimator` stores
		`segm_decoder_channels`, `segm_decoder_block_depth`, `segm_decoder_block_kernel`, and
		`segm_decoder_block_expansion`.
	upsample and shared layer settings : forwarded parameter family
		See `hunterFormsBS.hyperACE.ProgressiveUpsampleHead.__init__` [6].
		`ParametersMaskEstimator` stores `segm_upsample_scales`, `segm_upsample_tfc_tdf_depth`,
		`segm_upsample_tfc_tdf_bn`, `segm_activation`, `segm_norm_eps`, `segm_norm_affine`,
		`segm_conv_bias`, and `segm_linear_bias`.

	References
	----------
	[1] `hunterFormsBS.bandSplit.MaskEstimator`

	[2] `hunterFormsBS.hyperACE.SegmModel.__init__`

	[3] `hunterFormsBS.hyperACE.Backbone.__init__`

	[4] `hunterFormsBS.hyperACE.HyperACE.__init__`

	[5] `hunterFormsBS.hyperACE.Decoder.__init__`

	[6] `hunterFormsBS.hyperACE.ProgressiveUpsampleHead.__init__`
	"""
	activation: type[nn.Module]
	dim: int
	depth: int
	mlp_expansion_factor: int
	segm_out_bins: int | None
	segm_out_channels: int
	segm_base_channels: int
	segm_base_depth: int
	segm_num_hyperedges: int
	segm_num_heads: int
	segm_backbone_channels: tuple[int, int, int, int, int] | None
	segm_hyperace_k: int
	segm_hyperace_l: int
	segm_hyperace_c_h: float
	segm_hyperace_c_l: float
	segm_hyperace_c3ah_expansion: float
	segm_hyperace_low_order_depth: int
	segm_hyperace_low_order_kernel: int
	segm_hyperace_low_order_expansion: float
	segm_hyperace_out_channels: int | None
	segm_decoder_channels: list[int] | tuple[int, int, int, int] | None
	segm_decoder_block_depth: int
	segm_decoder_block_kernel: int
	segm_decoder_block_expansion: float
	segm_upsample_scales: tuple[int, int, int, int]
	segm_upsample_tfc_tdf_depth: int
	segm_upsample_tfc_tdf_bn: int
	segm_activation: type[nn.Module]
	segm_norm_eps: float
	segm_norm_affine: bool
	segm_conv_bias: bool
	segm_linear_bias: bool
	use_hyperACE: bool

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
	separator models forward stable field names to the shared attention stack. The record stores
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
	linear_attn : bool
		Legacy compatibility flag preserved for older configuration files. The current shared
		transformer stack keeps only the standard attention path, so `linear_attn` is forwarded as
		metadata rather than selecting a separate linear-attention implementation.
	norm_output : bool
		Whether to normalize after the last layer.
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
	linear_attn: bool
	norm_output: bool
	sage_attention: bool
	scale: float | None
