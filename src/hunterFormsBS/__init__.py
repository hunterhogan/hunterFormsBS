"""Access typed band-split source-separation models and supporting modules.

You can use this package to assemble typed source-separation models for music and other audio
mixtures. You can import the primary separator model, attention building blocks, band-partitioning
layers, mask estimation utilities, loss computation utilities, and typed configuration records from
the top-level package namespace.

Modules
-------
attend
	Attention layers and Transformer blocks for separator models.
bandSplit
	Band partitioning layers, mask estimation, and loss computation utilities.
bandSplitRotator
	Primary band-split separator model exposed by the package.
bs_roformer
	Compatibility implementation that preserves an upstream band-split separator layout.
mel_band_roformer
	Compatibility implementation that preserves an upstream mel-band separator layout.
make_static_mask_filter_bank
	Ad-hoc helpers that print static `mask_filter_bank` source for custom band layouts.
theTypes
	Typed configuration records for attention, STFT, transformer, and loss setup.
"""

from __future__ import annotations

from hunterFormsBS.bandSplitRotator import BandSplitRotator as BandSplitRotator
