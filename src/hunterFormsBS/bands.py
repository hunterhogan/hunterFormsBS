"""Build the frequency-domain front end for band-split source separation.

You can use this module to partition STFT frequency bins into bands and project each band into a
common feature width. `BandSplit` is the shared front-end projector consumed by
`hunterFormsBS.bandSplitRotator.BandSplitRotator` [1][2][3]. `DEFAULT_FREQS_PER_BANDS` provides
the standard non-overlapping frequency-bin partition used by the BS-RoFormer front end.

Contents
--------
Classes
	BandSplit
		Project concatenated band slices to a shared feature width.

Variables
	DEFAULT_FREQS_PER_BANDS
		Standard non-overlapping frequency-bin partition for the BS-RoFormer front end.

References
----------
[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
	Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
[2] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
	Transcription https://arxiv.org/abs/2409.04702
[3] `hunterFormsBS.bandSplitRotator.BandSplitRotator`
"""
from __future__ import annotations

from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch_einops_kit.scaleValues import RMSNorm
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
	from collections.abc import Sequence

DEFAULT_FREQS_PER_BANDS: tuple[int, ...] = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)

class BandSplit(Module):
	"""Project band slices to a shared feature width.

	You can use this class to convert one concatenated band representation into one stack of band
	tokens with a common feature width. `BandSplit` treats the last axis of the input `Tensor` as
	consecutive band slices, applies one learned projection module to each slice, and returns one
	`Tensor` with an explicit band axis. `BandSplit` is the shared front-end projector used by
	`hunterFormsBS.bandSplitRotator.BandSplitRotator` [1].

	Attributes
	----------
	dim_inputs : list[int]
		Width of each contiguous slice consumed from the last axis of the input `Tensor`.
	to_features : ModuleList
		Per-band projection module collection. Each entry transforms one slice width from `dim_inputs`
		to the shared feature width configured in `__init__`.

	References
	----------
	[1] `hunterFormsBS.bandSplitRotator.BandSplitRotator`
	"""
	def __init__(self, dim: int, dim_inputs: Sequence[int]) -> None:
		"""Configure one projection module per input band.

		You can use `__init__` to declare how `BandSplit` should partition the last axis of each input
		`Tensor` and how wide each projected band token should become. `__init__` stores the partition
		widths in `self.dim_inputs` and builds one projection module for each partition.

		Parameters
		----------
		dim : int
			Shared feature width produced for every band slice.
		dim_inputs : Sequence[int]
			Width sequence that partitions the last axis of the input `Tensor` into contiguous band
			slices.
		"""
		super().__init__()
		self.dim_inputs: list[int] = list(dim_inputs)
		self.to_features: ModuleList = ModuleList([])

		for dim_in in self.dim_inputs:
			net: nn.Sequential = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
			self.to_features.append(net)

	def forward(self, x: Tensor) -> Tensor:
		"""Project each band slice in `x` to one shared feature width.

		You can use `forward` when `x` already stores one concatenated slice per band along the last
		axis. `forward` splits `x` according to `self.dim_inputs`, applies the learned projection
		module associated with each slice, and returns one `Tensor` with one explicit band axis.

		Parameters
		----------
		x : Tensor
			Input band representation. `x.shape[-1]` must equal `sum(self.dim_inputs)`.

		Returns
		-------
		band_features : Tensor
			Projected band representation. `band_features.shape[-2]` equals `len(self.dim_inputs)`.
			`band_features.shape[-1]` equals the shared projection width configured in `__init__`.

		Shape Transformation
		--------------------
		input partition : relation
			`forward` partitions `x` into contiguous last-axis slices whose widths come from
			`self.dim_inputs`.
		output layout : relation
			If the leading shape of `x` is `leadingShape`, the return value has shape `(*leadingShape,
			len(self.dim_inputs), projectedWidth)`, where `projectedWidth` is the shared projection
			width configured in `__init__`.
		"""
		tuple_inputs: tuple[Tensor, ...] = torch.split(x, self.dim_inputs, dim=-1)
		# TODO why is dim hardcoded?

		outs: list[Tensor] = []
		for split_input, to_feature in zip(tuple_inputs, self.to_features, strict=True):
			split_output: Tensor = to_feature(split_input)
			outs.append(split_output)

		return torch.stack(outs, dim=-2)
