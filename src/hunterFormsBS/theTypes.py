"""Define shared typed data structures for the hunterFormsBS source-separation models.

You can use this module to access the typed containers that carry configuration across module
boundaries in hunterFormsBS. Each type is a `TypedDict` or `NamedTuple` that groups related keyword
arguments so callers can construct and pass them as a single named record rather than as separate
positional arguments. The types are consumed by `BandSplitRotator` [1], `BandSplit` [2],
`lossComputation` [2], and the attention stack in `attend` [3].

Contents
--------
Classes
    ParametersComputeLoss
        Collect multi-resolution STFT loss settings passed to `lossComputation`.
    FlashAttentionConfig
        Store PyTorch scaled-dot-product-attention backend flags for `Attend`.
    ParametersAttention
        Collect shared keyword arguments for one attention-block configuration.
    ParametersSTFT
        Collect shared keyword arguments for forward and inverse STFT calls.
    ParametersTransformer
        Collect shared keyword arguments for one transformer-block configuration.

References
----------
[1] hunterFormsBS.bandSplitRotator.BandSplitRotator

[2] hunterFormsBS.bandSplit

[3] hunterFormsBS.attend
"""
from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable
	from PoPE_pytorch import PoPE
	from rotary_embedding_torch import RotaryEmbedding
	from torch import Tensor

class FlashAttentionConfig(NamedTuple):
	"""Store scaled-dot-product-attention backend flags.

	You can use `FlashAttentionConfig` to keep the backend switches together.

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
	"""Collect shared attention keyword arguments.

	You can use `ParametersAttention` to keep one attention-block configuration together.

	Attributes
	----------
	dim_head : int
		Size of the per-head feature ***dim***ension.
	dim : int
		Size of the model feature ***dim***ension.
	attn_dropout : float
		Dropout probability inside the attention block.
	flash : bool
		Whether the attention block may use the flash path.
	heads : int
		Number of attention heads.
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
	"""Collect multi-resolution STFT loss settings.

	You can use `ParametersComputeLoss` to keep the shared STFT loss settings together.

	Attributes
	----------
	hop_length : int
		Hop size between adjacent STFT frames.
	loss_weight : float
		Multiplier for the multi-resolution STFT loss term.
	window_sizes : tuple[int, ...]
		Each STFT ***window*** size to evaluate.
	n_fft : int
		Minimum FFT size used for each STFT.
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
	"""Collect shared STFT keyword arguments.

	You can use `ParametersSTFT` to keep one STFT configuration together.

	Attributes
	----------
	hop_length : int
		Hop size between adjacent STFT frames.
	n_fft : int
		FFT size used by the STFT.
	normalized : bool
		Whether the STFT uses normalized scaling.
	win_length : int
		Length of the STFT ***win***dow.
	"""
	hop_length: int
	n_fft: int
	normalized: bool
	win_length: int

class ParametersTransformer(TypedDict):
	"""Collect shared transformer keyword arguments.

	You can use `ParametersTransformer` to keep one transformer-block configuration together.

	Attributes
	----------
	attn_dropout : float
		Dropout probability in the ***attn***ention sublayer.
	dim : int
		Size of the model feature ***dim***ension.
	dim_head : int
		Size of the per-head feature ***dim***ension.
	ff_dropout : float
		Dropout probability in the feedforward sublayer.
	flash_attn : bool
		Whether attention may use the flash-***attn***ention path.
	heads : int
		Number of attention heads.
	norm_output : bool
		Whether to normalize after the last layer.
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
