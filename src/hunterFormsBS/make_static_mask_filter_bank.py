"""Generate static `mask_filter_bank` source for custom band layouts.

You can use this module to print paste-ready Python source that defines a `torch.bool`
`mask_filter_bank` `Tensor` for `hunterFormsBS.bandSplitRotator.BandSplitRotator` [2] or any other
`class` with a `mask_filter_bank` parameter [1]. The module is intentionally separate from the model
code, imports `librosa` only inside `librosa_filters_mel` [4], and is meant for ad-hoc direct Python
calls when a checkpoint uses a custom band layout.

This module does not install a command-line interface. Import the function you need from Python and
call the function from a REPL, notebook, or one-off script. Most users never need this module because
the package already bundles the common lucidrains-style mel-band split in
`hunterFormsBS.bandSplit.mask_filter_bank_mel_band_default` [3].

Contents
--------
Functions
    filter_bank_non_overlapping
        Generate one static non-overlapping band split from contiguous band widths.
    librosa_filters_mel
        Generate one static mel-band band split with `librosa.filters.mel`.
    print_static_mask
        Print one paste-ready `torch.bool` assignment for `mask_filter_bank`.

References
----------
[1] hunterFormsBS.mel_band_roformer.MelBandRoformer

[2] hunterFormsBS.bandSplitRotator.BandSplitRotator

[3] hunterFormsBS.bandSplit.mask_filter_bank_mel_band_default

[4] librosa.filters.mel - librosa
    https://librosa.org/doc/main/generated/librosa.filters.mel.html
"""

from __future__ import annotations

from hunterMakesPy import raiseIfNone
from hunterMakesPy.dataStructures import autoDecodingRLE
from typing import Any, TYPE_CHECKING
import numpy
import numpy.typing
import sys

if TYPE_CHECKING:
	from collections.abc import Sequence

def filter_bank_non_overlapping(freqs_per_bands: Sequence[int], num_bands: int | None = None) -> None:
	"""Generate one static non-overlapping `mask_filter_bank` from `freqs_per_bands`.

	You can use this function to create one Boolean band-membership matrix for a non-overlapping
	layout and print one paste-ready Python assignment through `print_static_mask` [1]. The function
	expands each band index according to `freqs_per_bands`, compares the expanded index vector
	against each band index, and writes the resulting `mask_filter_bank` source to standard output.

	Parameters
	----------
	freqs_per_bands : Sequence[int]
		Contiguous frequency-bin counts for each band in ascending band order.
	num_bands : int | None = None
		Optional band-count override. When `num_bands` is `None`, the function uses
		`len(freqs_per_bands)`.

	Standard Output
	---------------
	generated_source : str
		The function writes one `torch.tensor(dtype=torch.bool, data=...)` assignment that you can
		paste into Python source as a static `mask_filter_bank` definition.

	See Also
	--------
	print_static_mask
		Print the compact `torch` source representation used by this module.

	References
	----------
	[1] hunterFormsBS.make_static_mask_filter_bank.print_static_mask
	"""
	num_bands = num_bands or len(freqs_per_bands)
	filter_bank: numpy.ndarray[tuple[int], numpy.dtype[numpy.int_]] = numpy.repeat(numpy.arange(num_bands), freqs_per_bands)
	mask_filter_bank: numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool_]] = filter_bank[None, :] == numpy.arange(num_bands)[:, None]
	print_static_mask(mask_filter_bank)

def librosa_filters_mel(
	sample_rate: int | None = None,
	stft_n_fft: int | None = None,
	num_bands: int | None = None,
	sr: float | None = None,
	n_fft: int | None = None,
	n_mels: int | None = None,
	**keywordArguments: Any,
) -> None:
	"""Generate one static mel-band `mask_filter_bank` with `librosa.filters.mel` [1].

	You can use this function to create one Boolean mel-band band-membership matrix and print one
	paste-ready Python assignment through `print_static_mask` [2]. The function keeps `librosa`
	optional for the package because the `librosa` import happens only when `librosa_filters_mel`
	runs. The function also forces `mask_filter_bank[0, 0] = True` before printing so the DC bin
	remains explicitly covered, matching the packaged default mask family [3]. With
	`sample_rate=44100`, `stft_n_fft=2048`, and `num_bands=60`, the Boolean mask matches the common
	bundled mel-band default [3].

	Parameters
	----------
	sample_rate : int | None = None
		Sample-rate alias for `sr`.
	stft_n_fft : int | None = None
		FFT-size alias for `n_fft`.
	num_bands : int | None = None
		Mel-band-count alias for `n_mels`.
	sr : float | None = None
		Sample rate passed to `librosa.filters.mel` when `sample_rate` is not supplied.
	n_fft : int | None = None
		FFT size passed to `librosa.filters.mel` when `stft_n_fft` is not supplied.
	n_mels : int | None = None
		Mel-band count passed to `librosa.filters.mel` when `num_bands` is not supplied.

	Other Parameters
	----------------
	**keywordArguments : Any
		Additional keyword arguments forwarded unchanged to `librosa.filters.mel`.

	Standard Output
	---------------
	generated_source : str
		The function writes one `torch.tensor(dtype=torch.bool, data=...)` assignment that you can
		paste into Python source as a static `mask_filter_bank` definition.

	Dependency Behavior
	-------------------
	optional dependency import : behavior
		`librosa` is imported only inside `librosa_filters_mel`. Importing
		`hunterFormsBS.make_static_mask_filter_bank` does not require `librosa`.

	Parameter Aliases
	-----------------
	alias pairs : relation
		`sample_rate` and `sr` name the same `librosa.filters.mel` parameter. `stft_n_fft` and
		`n_fft` name the same parameter. `num_bands` and `n_mels` name the same parameter.

	See Also
	--------
	print_static_mask
		Print the compact `torch` source representation used by this module.

	References
	----------
	[1] librosa.filters.mel - librosa
		https://librosa.org/doc/main/generated/librosa.filters.mel.html
	[2] hunterFormsBS.make_static_mask_filter_bank.print_static_mask

	[3] hunterFormsBS.bandSplit.mask_filter_bank_mel_band_default
	"""
	from librosa import filters  # noqa: PLC0415
	sr = raiseIfNone(sr or sample_rate)
	n_fft = raiseIfNone(n_fft or stft_n_fft)
	n_mels = raiseIfNone(n_mels or num_bands)
	filter_bank: numpy.typing.NDArray[numpy.float32] = filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, **keywordArguments)
	mask_filter_bank: numpy.typing.NDArray[numpy.bool] = filter_bank > 0
	mask_filter_bank[0, 0] = True
	print_static_mask(mask_filter_bank)

def print_static_mask(mask_filter_bank: numpy.typing.NDArray[numpy.bool]) -> None:
	"""Print one paste-ready static `mask_filter_bank` assignment.

	You can use this function to serialize one Boolean band-membership matrix to compact Python
	source for later pasting into a module. The function compresses `mask_filter_bank` before
	printing and writes both the `import torch` line and the `mask_filter_bank` assignment to
	standard output.

	Parameters
	----------
	mask_filter_bank : numpy.typing.NDArray[numpy.bool]
		Boolean band-membership matrix with shape `(band, freq)`.

	Standard Output
	---------------
	generated_source : str
		The function writes Python source that recreates `mask_filter_bank` as one
		`torch.tensor(dtype=torch.bool, data=...)` value.
	"""
	rle: str = autoDecodingRLE(mask_filter_bank.astype(int))

	sys.stdout.write('import torch\n')
	sys.stdout.write(f'mask_filter_bank: torch.Tensor = torch.tensor(dtype=torch.bool, data={rle})\n')
