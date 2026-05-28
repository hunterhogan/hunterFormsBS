"""Access typed band-split source-separation modules and the primary separator.

You can use this package to assemble typed source-separation models for music and other audio
mixtures. You can import the primary separator model from the top-level package namespace and reach
the supporting attention, band-partitioning, optional segmentation-branch, and typed-configuration
modules through `hunterFormsBS.*` submodule imports.

Modules
-------
attend
	Attention layers for separator models.
bands
	Band projection layer and the BS-style default frequency-bin partition.
bandSplitRotator
	Primary band-split separator model exposed by the package.
hyperACE
	Optional segmentation-style mask-estimation branch and supporting building blocks.
loss
	Multi-resolution STFT training-loss computation utilities.
mask
	Mask estimation heads and band-local affine blocks.
theTypes
	Typed configuration records for attention, mask estimation, STFT, transformer, and loss setup.
transform
	Position-wise feedforward block and stacked attention-and-feedforward transformer.
"""

from __future__ import annotations

from hunterFormsBS.bandSplitRotator import BandSplitRotator as BandSplitRotator
