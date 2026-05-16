"""Use the Mel-Band RoFormer waveform separator.

You can use this module to instantiate one configurable waveform-to-waveform music source separator
that follows the overlapped mel-band front end from Mel-Band RoFormer [1]. The module exposes one
public class with a forward method suitable for both inference and supervised training.

For the common lucidrains-style mel-band layout, this module uses the bundled
`hunterFormsBS.bandSplit.mask_filter_bank_mel_band_default` when `mask_filter_bank` is `None`,
`sample_rate=44100`, `stft_n_fft=2048`, and `num_bands=60`. If a checkpoint uses a different
mel-band layout, pass `mask_filter_bank` explicitly. You can generate a static custom value with
`hunterFormsBS.make_static_mask_filter_bank.librosa_filters_mel`.

Contents
--------
Classes
	MelBandRoformer
		Waveform separator with overlapped mel-band front end, RoPE or PoPE positional embedding,
		optional linear-attention pre-block, and multi-resolution STFT training loss.

References
----------
[1] Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation.
	https://arxiv.org/abs/2409.04702
"""
from __future__ import annotations

from einops import pack, rearrange, reduce, repeat, unpack  # pyright: ignore[reportUnknownVariableType]
from functools import partial
from hunterFormsBS.attend import Transformer
from hunterFormsBS.bandSplit import BandSplit, DEFAULT_FREQS_PER_BANDS, lossComputation, mask_filter_bank_mel_band_default, MaskEstimator
from hunterFormsBS.theTypes import ComputeLoss, KwargsSTFT, KwargsTransformer
from hunterMakesPy import raiseIfNone
from more_itertools import loops
from operator import mul
from PoPE_pytorch import PoPE
from rotary_embedding_torch import RotaryEmbedding
from torch import nn, tensor, Tensor
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportUnknownVariableType]
from torch_einops_kit import exists
from torch_einops_kit.einops import pack_one, unpack_one
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast, TYPE_CHECKING
from Z0Z_tools import halfsineTensor
import torch

if TYPE_CHECKING:
	from collections.abc import Callable

class MelBandRoformer(Module):
	"""Separate a raw-audio mixture into stem audio with Mel-Band RoFormer.

	You can use this class to instantiate one waveform-to-waveform separator that accepts raw-audio
	`Tensor` `raw_audio`, converts waveform `Tensor` `raw_audio` to one complex short-time Fourier
	transform (STFT) representation, groups frequency bins into band tokens, predicts one complex mask
	for each requested stem, and returns reconstructed stem waveform `Tensor` output or one training
	loss through `hunterFormsBS.bandSplit.lossComputation` [5]. The class follows the overlapped bands
	front end from Mel-Band RoFormer [2], but can use any band configuration, not just mel bands.

	The class also lets you switch between RoPE (rotary position embedding) [3] and PoPE (polar
	coordinate positional embedding) [4], insert one optional linear-attention block before the time
	attention block and frequency attention block, and keep compatibility controls that are practical
	when porting configuration files or training loops from Music-Source-Separation-Training
	[6][7][8].

	Attributes
	----------
	audio_channels : int
		Number of waveform channels expected at the input and reconstructed at the output.
	band_split : Module
		Front-end projection module that converts gathered STFT band slices to model-width band token
		`Tensor` `x`.
	final_norm : Module
		Final normalization module applied to band token `Tensor` `x` after the hierarchical attention
		stack.
	freq_indices : Tensor
		Gather index map that lists the frequency members belonging to each band. `forward` uses
		`freq_indices` to gather band-local STFT slices.
	linear_transformer_depth : int
		Depth of the optional linear-attention block inserted before the time attention block and the
		frequency attention block inside each hierarchical layer.
	layers : ModuleList
		Hierarchical attention stack. Each entry stores an optional linear-attention block followed by
		the time attention block and the frequency attention block.
	mask_filter_bank : Tensor
		Band-membership map registered as a non-persistent buffer. Entry `(bandIndex, frequencyIndex)`
		is `True` when that frequency bin belongs to that band.
	mask_estimators : ModuleList
		One mask-estimator head per configured output source estimate. Each head predicts one complex
		mask from band token `Tensor` `x`.
	match_input_audio_length : bool
		Whether inverse STFT reconstruction is forced back to the input waveform length.
	multi_stft : ComputeLoss
		Internal multi-resolution STFT loss configuration consumed only when `forward` receives
		`target` [11].
	num_bands_per_freq : Tensor
		Overlap count per frequency bin in `mask_filter_bank`. The count lets `forward` average
		repeated complex-mask estimates when one frequency bin belongs to more than one band.
	num_freqs_per_band : Tensor
		Frequency-bin count for each band in `mask_filter_bank`. The count sizes the band front end
		and each mask-estimator head.
	num_stems : int
		Number of configured output sources.
	skip_connection : bool
		Whether later hierarchical blocks add stored band token `Tensor` output from earlier
		hierarchical blocks.
	stereo : bool
		Whether `forward` expects stereo waveform `Tensor` `raw_audio` instead of mono waveform
		`Tensor` `raw_audio`.
	stft_kwargs : KwargsSTFT
		Keyword record shared by the forward and inverse STFT calls [12].
	stft_window_fn : Callable[..., Tensor]
		Partially applied window constructor used by the forward and inverse STFT calls.
	use_torch_checkpoint : bool
		Whether selected modules are executed through activation checkpointing [13].
	zero_dc : bool
		Whether the DC (zero-frequency) bin is zeroed before inverse STFT reconstruction.

	Configuration modes
	-------------------
	automatic front-end selection : behavior
		When `mask_filter_bank` is `None`, `sample_rate` together with `num_bands` requests the
		overlapped mel-band front end from `MelBandRoFormer` [2].
	effective default selection : behavior
		`final_norm`, `norm_output`, and `zero_dc` accept `None` only to preserve historical behavior
		from configuration files made for `BSRoFormer` [9] and `MelBandRoFormer` [10]. In mel-band
		mode the effective defaults are `final_norm=False`, `norm_output=True`, and `zero_dc=False`.
		In non-overlapping band-split mode the effective defaults are `final_norm=True`,
		`norm_output=False`, and `zero_dc=True`. New configuration files are easier to reason about
		when all three values are set explicitly.

	Framework adaptation
	--------------------
	stem selection : behavior
		`forward` accepts `active_stem_ids`. The selected mask-estimator head index list becomes the
		`stem_ids` value passed to `hunterFormsBS.bandSplit.lossComputation` [5], so an external
		training loop can request only the target stem present in the current batch [8].
	checkpointing : behavior
		`use_torch_checkpoint` wraps `self.band_split`, the optional linear-attention block, the time
		attention block, the frequency attention block, and each mask-estimator head with activation
		checkpointing [13]. This behavior is practical for memory-constrained training in external
		frameworks [6][7][8], but it is not a defining architectural requirement from the model papers
		[1][2].

	See Also
	--------
	hunterFormsBS.bs_roformer.BSRoformer
		Specialized non-overlapping band-split `class`.
	hunterFormsBS.bandSplitRotator.BandSplitRotator
		Unified separator that uses `BandSplit` as the band front end.

	References
	----------
	[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
		Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
	[2] Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation.
		https://arxiv.org/abs/2409.04702
	[3] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
		Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
	[4] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., & Mozer, M. C. (2025). Decoupling the
		"What" and "Where" With Polar Coordinate Positional Embeddings.
		https://arxiv.org/abs/2509.10534
	[5] hunterFormsBS.bandSplit.lossComputation

	[6] ZFTurbo/Music-Source-Separation-Training
		https://github.com/ZFTurbo/Music-Source-Separation-Training
	[7] ZFTurbo/Music-Source-Separation-Training/utils/settings.py
		https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/utils/settings.py
	[8] ZFTurbo/Music-Source-Separation-Training/train.py
		https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/train.py
	[9] hunterFormsBS.bs_roformer.BSRoformer

	[10] hunterFormsBS.mel_band_roformer.MelBandRoformer

	[11] hunterFormsBS.theTypes.ComputeLoss

	[12] hunterFormsBS.theTypes.KwargsSTFT

	[13] torch.utils.checkpoint.checkpoint
		https://pytorch.org/docs/stable/checkpoint.html
	"""
	def __init__(
		self,
		dim: int,
		*,
		attn_dropout: float = 0.0,
		depth: int,
		dim_freqs_in: int = 1025,  # noqa: ARG002
		dim_head: int = 64,
		ff_dropout: float = 0.0,
		final_norm: bool | None = None,
		flash_attn: bool = True,
		freq_transformer_depth: int = 2,
		freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # noqa: ARG002
		heads: int = 8,
		linear_transformer_depth: int = 0,  # noqa: ARG002
		mask_estimator_depth: int | None = None,
		mask_filter_bank: Tensor | None = None,
		match_input_audio_length: bool = True,
		mlp_expansion_factor: int = 4,
		multi_stft_hop_size: int = 147,
		multi_stft_normalized: bool = False,
		multi_stft_resolution_loss_weight: float = 1.0,
		multi_stft_resolutions_window_sizes: tuple[int, ...] = (4096, 2048, 1024, 512, 256),
		multi_stft_window_fn: Callable[..., Tensor] = halfsineTensor,
		norm_output: bool = False,
		num_bands: int | None = None,
		num_stems: int = 1,
		sample_rate: float | None = None,
		skip_connection: bool = False,
		stereo: bool = False,
		stft_hop_length: int = 512,
		stft_n_fft: int = 2048,
		stft_normalized: bool = False,
		stft_win_length: int = 1024,
		stft_window_fn: Callable[..., Tensor] = halfsineTensor,
		time_transformer_depth: int = 2,
		use_pope: bool = False,
		use_torch_checkpoint: bool = False,
		zero_dc: bool | None = None,
	) -> None:
		"""Configure the unified RoFormer variant and its band front end.

		You can use `__init__` to choose the band layout, the hierarchical attention layout, the
		forward and inverse short-time Fourier transform (STFT) settings, the internal
		multi-resolution STFT loss settings, and the compatibility switches for the unified separator.
		`__init__` accepts explicit mode values for new configuration files and still accepts `None`
		for several toggles so older configuration files keep their historical behavior [8][9].
		`__init__` also chooses whether the non-linear attention blocks use RoPE (rotary position
		embedding) [3] or PoPE (polar coordinate positional embedding) [4].

		Parameters
		----------
		dim : int
			Feature width of each band token after the front-end projection.
		attn_dropout : float = 0.0
			Dropout probability inside each attention block in the hierarchical Transformer stack.
		depth : int
			Number of top-level layer groups. Each group contains the time attention block, the
			frequency attention block, and optionally one linear-attention block.
		dim_freqs_in : int = 1025
			Compatibility field kept for external configuration files. The current constructor derives
			the effective frequency-bin layout from `stft_n_fft`, `mask_filter_bank`,
			`freqs_per_bands`, and the automatic front-end selection logic instead of reading
			`dim_freqs_in`.
		dim_head : int = 64
			Per-head feature width for the non-linear time attention block and frequency attention
			block.
		ff_dropout : float = 0.0
			Dropout probability inside each feedforward block.
		final_norm : bool | None = None
			Toggle the final normalization module. `None` is a compatibility sentinel for
			configuration files that relied on the historical defaults of `BSRoformer` [8] and
			`MelBandRoformer` [9], not a distinct modeling mode. The effective default is `True` in
			non-overlapping band-split mode and `False` in mel-band mode. New configuration files
			should set `final_norm` explicitly.
		flash_attn : bool = True
			Request the flash-attention path inside each attention block when the underlying attention
			implementation supports that path.
		freq_transformer_depth : int = 2
			Depth of the per-layer frequency attention block.
		freqs_per_bands : tuple[int, ...] = DEFAULT_FREQS_PER_BANDS
			Non-overlapping frequency-bin counts used when the constructor builds the BS-style band
			front end.
		heads : int = 8
			Number of attention heads in each non-linear attention block.
		linear_transformer_depth : int = 0
			Depth of the optional linear-attention block inserted before the time attention block and
			the frequency attention block inside each hierarchical layer.
		mask_estimator_depth : int | None = None
			Depth of the per-band MLP inside each mask-estimator head.
		mask_filter_bank : Tensor | None = None
			Custom band-membership `Tensor` with shape `(band, freq)`. Entry `(bandIndex,
			frequencyIndex)` is truthy when that frequency bin belongs to that band. When
			`mask_filter_bank` is provided, `__init__` skips automatic BS-mode or mel-mode band
			construction. For ad-hoc custom generation,
			`hunterFormsBS.make_static_mask_filter_bank.librosa_filters_mel` prints one paste-ready
			static definition.
		match_input_audio_length : bool = True
			When `True`, inverse STFT reconstruction is forced back to the original waveform length.
		mlp_expansion_factor : int = 4
			Hidden-width expansion factor inside each mask-estimator MLP.
		multi_stft_hop_size : int = 147
			Hop size for the internal multi-resolution STFT loss.
		multi_stft_normalized : bool = False
			Normalization flag for the internal multi-resolution STFT transforms.
		multi_stft_resolution_loss_weight : float = 1.0
			Weight applied to the multi-resolution STFT term before `forward` adds that term to the
			waveform-domain loss.
		multi_stft_resolutions_window_sizes : tuple[int, ...] = (4096, 2048, 1024, 512, 256)
			Window size sequence used by the internal multi-resolution STFT loss.
		multi_stft_window_fn : Callable[..., Tensor] = halfsineTensor
			Window constructor used by the internal multi-resolution STFT loss.
		norm_output : bool | None = None
			Output-normalization toggle inside each attention block. `None` is a compatibility
			sentinel for wrapper-style configuration files that depended on historical mode-specific
			defaults [8][9]. The effective default is `False` in non-overlapping band-split mode and
			`True` in mel-band mode. New configuration files should set `norm_output` explicitly.
		num_bands : int | None = None
			Number of bands for automatic front-end construction. When `mask_filter_bank` is omitted,
			`sample_rate` together with `num_bands` selects mel-band mode. When `mask_filter_bank` is
			provided and `num_bands` is `None`, `__init__` infers `num_bands` from
			`mask_filter_bank.shape[0]`. When non-overlapping band-split mode is active, `__init__`
			does not silently correct mismatches between `num_bands` and `freqs_per_bands`. The
			mismatch is preserved so configuration or checkpoint-loading errors fail quickly at the
			actual shape boundary.
		num_stems : int = 1
			Number of configured output sources.
		sample_rate : float | None = None
			Sample rate value used only when automatic mel-band construction is active. When
			`mask_filter_bank` is `None`, `sample_rate` together with `num_bands` selects mel-band
			mode.
		skip_connection : bool = False
			If `True`, each hierarchical layer adds stored band token `Tensor` output from every
			earlier hierarchical layer before the current layer runs.
		stereo : bool = False
			Expect stereo waveform input and reconstruct stereo waveform output. When `stereo` is
			`False`, the class expects mono waveform input.
		stft_hop_length : int = 512
			Hop length for the forward and inverse STFT.
		stft_n_fft : int = 2048
			FFT size for the forward and inverse STFT.
		stft_normalized : bool = False
			Normalization flag for the forward and inverse STFT.
		stft_win_length : int = 1024
			Length of the windowing function for the forward and inverse STFT.
		stft_window_fn : Callable[..., Tensor] = halfsineTensor
			Callable Python function to construct the windowing function used by the forward and
			inverse STFT.
		time_transformer_depth : int = 2
			Depth of the per-layer time attention block.
		use_pope : bool = False
			Replace RoPE (rotary position embedding) [3] with PoPE (polar coordinate positional
			embedding) [4] in the time attention block and frequency attention block.
		use_torch_checkpoint : bool = False
			Enable activation checkpointing for `self.band_split`, each attention block, and each
			mask-estimator head. This option exists to ease memory-constrained training in external
			frameworks such as Music-Source-Separation-Training [5][6][7].
		zero_dc : bool | None = None
			Toggle zeroing of the DC (zero-frequency) bin before inverse STFT reconstruction. `None`
			is a compatibility sentinel for configuration files that relied on historical
			mode-specific defaults [8][9]. The effective default is `True` in non-overlapping
			band-split mode and `False` in mel-band mode. New configuration files should set `zero_dc`
			explicitly.

		References
		----------
		[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
			Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
		[2] Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation.
			https://arxiv.org/abs/2409.04702
		[3] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
			Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
		[4] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., & Mozer, M. C. (2025). Decoupling the
			"What" and "Where" With Polar Coordinate Positional Embeddings.
			https://arxiv.org/abs/2509.10534
		[5] ZFTurbo/Music-Source-Separation-Training
			https://github.com/ZFTurbo/Music-Source-Separation-Training
		[6] ZFTurbo/Music-Source-Separation-Training/utils/settings.py
			https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/utils/settings.py
		[7] ZFTurbo/Music-Source-Separation-Training/train.py
			https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/train.py
		[8] hunterFormsBS.bs_roformer.BSRoformer

		[9] hunterFormsBS.mel_band_roformer.MelBandRoformer
		"""
		super().__init__()

		# class attributes, including "forward" compatibility

		self.stereo: bool = stereo
		self.audio_channels: int = 2 if self.stereo else 1
		self.num_stems: int = num_stems
		self.use_torch_checkpoint: bool = use_torch_checkpoint
		self.skip_connection: bool = skip_connection

		if mask_filter_bank is None:
			num_bands = num_bands or 60
			sample_rate = sample_rate or 44100
			if (stft_n_fft == 2048) and (num_bands == 60) and (sample_rate == 44100):
				mask_filter_bank = mask_filter_bank_mel_band_default
			else:
				message: str = (
					f'I received `{stft_n_fft = }`, `{num_bands = }`, and `{sample_rate = }`, but '
					'I only provide one built-in mel-band `mask_filter_bank` when `mask_filter_bank` '
					'is `None`: `stft_n_fft == 2048`, `num_bands == 60`, and `sample_rate == 44100`. '
					'If your checkpoint uses a different mel-band split, pass `mask_filter_bank` '
					'explicitly. You can generate a static `mask_filter_bank` value with '
					'`hunterFormsBS.make_static_mask_filter_bank.librosa_filters_mel`.'
				)
				raise ValueError(message)
			mask_estimator_depth = mask_estimator_depth or 1
			if final_norm is None:
				final_norm = False
			if zero_dc is None:
				zero_dc = False

		self.zero_dc: bool = raiseIfNone(zero_dc, f'I received {zero_dc = }, but I need a type `bool` value or a "truthy" value.')
		self.final_norm: RMSNorm | nn.Identity = RMSNorm(dim) if raiseIfNone(final_norm, f'I received {final_norm = }, but I need a type `bool` value or a "truthy" value.') else nn.Identity()

		# rotator and transformer

		if use_pope:
			time_pope_embed: PoPE | None = PoPE(dim=dim_head, heads=heads)
			freq_pope_embed: PoPE | None = PoPE(dim=dim_head, heads=heads)
			time_rotary_embed: RotaryEmbedding | None = None
			freq_rotary_embed: RotaryEmbedding | None = None
		else:
			time_rotary_embed = RotaryEmbedding(dim=dim_head)
			freq_rotary_embed = RotaryEmbedding(dim=dim_head)
			time_pope_embed = None
			freq_pope_embed = None

		transformer_kwargs: KwargsTransformer = KwargsTransformer(attn_dropout=attn_dropout, dim_head=dim_head,
			dim=dim, ff_dropout=ff_dropout, flash_attn=flash_attn, heads=heads, norm_output=norm_output,
		)

		self.layers: ModuleList = ModuleList([])
		for _deep in loops(depth):
			self.layers.append(nn.ModuleList([
				Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, pope_embed=time_pope_embed, **transformer_kwargs)
				, Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, pope_embed=freq_pope_embed, **transformer_kwargs)
			]))

		# stft

		self.stft_kwargs: KwargsSTFT = KwargsSTFT(n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length, normalized=stft_normalized)

		self.stft_window_fn: Callable[..., Tensor] = partial(stft_window_fn, stft_win_length)

		self.match_input_audio_length: bool = match_input_audio_length

		# band split and mask estimator

		self.register_buffer('mask_filter_bank', mask_filter_bank, persistent=False)
		num_freqs_per_band: Tensor = reduce(mask_filter_bank, 'b f -> b', 'sum')
		self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)

		freqs_per_bands_with_complex: tuple[int, ...] = tuple(map(partial(mul, 2 * self.audio_channels), num_freqs_per_band.tolist())) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # ty:ignore[invalid-assignment]
		self.band_split: BandSplit = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
		self.mask_estimators: ModuleList = nn.ModuleList([])
		for _stem_index in loops(self.num_stems):
			self.mask_estimators.append(MaskEstimator(dim, freqs_per_bands_with_complex, depth=raiseIfNone(mask_estimator_depth, f'I received {mask_estimator_depth = }, but I need a type `int` > 0. If you are migrating a `BSRoformer` checkpoint and the old default value was `2`, then you probably want `mask_estimator_depth = 1` in this package.'), mlp_expansion_factor=mlp_expansion_factor))

		freqs: int = stft_n_fft // 2 + 1
		repeated_freq_indices: Tensor = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
		freq_indices: Tensor = repeated_freq_indices[mask_filter_bank]

		if self.stereo:
			freq_indices = repeat(freq_indices, 'f -> f s', s=2)
			freq_indices = freq_indices * 2 + torch.arange(2)
			freq_indices = rearrange(freq_indices, 'f s -> (f s)')

		self.register_buffer('freq_indices', freq_indices, persistent=False)
		num_bands_per_freq: Tensor = reduce(mask_filter_bank, 'b f -> f', 'sum')
		self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

		self.multi_stft: ComputeLoss = ComputeLoss(hop_length=multi_stft_hop_size, loss_weight=multi_stft_resolution_loss_weight,
			n_fft=stft_n_fft, normalized=multi_stft_normalized, window_fn=multi_stft_window_fn, window_sizes=multi_stft_resolutions_window_sizes,
		)

	def forward(self, raw_audio: Tensor, target: Tensor | None = None, active_stem_ids: list[int] | None = None, *, return_loss_breakdown: bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
		"""Separate `raw_audio` into stem waveform output or training loss.

		You can use `forward` for both inference and training. `forward` accepts mixture waveform
		`Tensor` `raw_audio`, converts waveform `Tensor` `raw_audio` to one complex short-time Fourier
		transform (STFT) representation `Tensor` `stft_repr`, gathers band-local slices into band
		token `Tensor` `x`, routes band token `Tensor` `x` through the unified `BSRoFormer` /
		`MelBandRoFormer` attention core [1][2], estimates one complex mask `Tensor` `masks` for each
		active stem, and reconstructs stem waveform `Tensor` `recon_audio` with the inverse STFT. If
		`Tensor` `target` is passed in, `forward` returns one total loss `Tensor` or one tuple
		containing the waveform-domain L1 term and the multi-resolution STFT term [5].

		The non-linear attention path uses RoPE (rotary position embedding) [3] or PoPE (polar
		coordinate positional embedding) [4] according to `__init__`. `active_stem_ids` lets one
		external training loop keep one multi-stem model while supervising only the stem present in
		the current batch [6][7], and `self.use_torch_checkpoint` lets the same call path trade
		compute for activation memory through activation checkpointing [8].

		Parameters
		----------
		raw_audio : Tensor
			Input mixture waveform `Tensor` `raw_audio`. `forward` accepts shape `(batch, time)` for
			mono shorthand or shape `(batch, channel, time)` for explicit channel layout. The
			`channel` axis must match `self.stereo`.
		target : Tensor | None = None
			Reference stem waveform `Tensor` `target` used only for loss computation. When `target` is
			`None`, `forward` returns reconstructed stem waveform `Tensor` `recon_audio`. When
			`target` exists, `forward` returns the loss from `hunterFormsBS.bandSplit.lossComputation`
			[5].
		active_stem_ids : list[int] | None = None
			Optional stem index list selecting which mask-estimator heads contribute to complex mask
			`Tensor` `masks` and which target stems `stem_ids` supervise [5]. When `active_stem_ids`
			is `None`, the default behavior is to use every configured head. This behavior is mainly
			intended for framework adapters such as Music-Source-Separation-Training [6][7].
		return_loss_breakdown : bool = False
			Request the expanded loss return path with both the waveform-domain term and the
			multi-resolution STFT term. `return_loss_breakdown` matters only when `target` exists.

		Returns
		-------
		recon_audio : Tensor
			`recon_audio` contains reconstructed stem waveform with axis order batch, stem, audio
			channel, time. Returned when `target` is `None`, the default behavior.
		total_loss : Tensor
			Returned when `target` exists and `return_loss_breakdown` is `False`.
		loss_with_breakdown : tuple[Tensor, tuple[Tensor, Tensor]]
			Returned when `target` exists and `return_loss_breakdown` is `True`. The outer `Tensor` is
			the total loss. The inner pair contains the waveform L1 term and the unweighted
			multi-resolution STFT loss term from `hunterFormsBS.bandSplit.lossComputation` [5].

		See Also
		--------
		hunterFormsBS.bandSplit.lossComputation
			Compute the waveform-domain loss and the multi-resolution STFT loss term.
		hunterFormsBS.bs_roformer.BSRoformer
			Specialized non-overlapping band-split `class`.
		hunterFormsBS.mel_band_roformer.MelBandRoformer
			Specialized overlapped mel-band `class`.

		Execution stages
		----------------
		spectral analysis and band gathering : behavior
			`forward` computes one complex STFT representation `Tensor` `stft_repr` from waveform
			`Tensor` `raw_audio`, converts the complex axis to one real-imaginary pair layout, merges
			the audio-channel axis into the frequency axis, and gathers the frequency members listed
			by `self.freq_indices`. The gathered view becomes band token `Tensor` `x`, where each time
			step carries the concatenated real and imaginary values for one band.
		hierarchical attention core : behavior
			Each layer applies the optional linear-attention block first, then the time attention
			block, then the frequency attention block. `use_pope` configured in `__init__` decides
			whether the non-linear attention block receives RoPE [3] or PoPE [4].
		mask accumulation : behavior
			Each active mask-estimator head predicts one complex mask `Tensor` `masks` over the
			gathered band view. When the front end uses overlapped mel bands [2], `forward`
			scatter-adds repeated frequency-bin estimates back to the full frequency layout and
			divides by `self.num_bands_per_freq`, so overlapped frequency bins receive the average
			complex mask instead of the sum.
		waveform reconstruction : behavior
			`forward` multiplies the original complex STFT representation by complex mask `Tensor`
			`masks`, applies the optional DC (zero-frequency) bin zeroing rule, and reconstructs
			waveform `Tensor` `recon_audio` with the inverse STFT. When
			`self.match_input_audio_length` is `True`, the inverse STFT is asked to match the original
			sample count.

		Framework adaptation
		--------------------
		activation checkpointing : behavior
			When `self.use_torch_checkpoint` is `True`, `forward` wraps `self.band_split`, the
			optional linear-attention block, the time attention block, the frequency attention block,
			and each mask-estimator head with activation checkpointing [8]. Activation checkpointing
			recomputes intermediate activations during the backward pass, which lowers activation
			memory use at the cost of extra compute [8].
		stem-target interoperability : behavior
			`active_stem_ids` exists so one external training loop can keep one multi-stem model while
			still computing loss only for the stem target present in the current batch [6][7].
			`forward` preserves the selected `stem_ids` list and passes `stem_ids` to
			`hunterFormsBS.bandSplit.lossComputation` [5].

		Device fallback
		---------------
		MPS retry path : behavior
			On `device.type == 'mps'`, where MPS means Apple Metal Performance Shaders, `forward`
			retries the forward STFT and inverse STFT on CPU when the direct call fails.

		References
		----------
		[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
			Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
		[2] Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source
			Separation. https://arxiv.org/abs/2409.04702
		[3] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
			Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
		[4] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., & Mozer, M. C. (2025). Decoupling
			the "What" and "Where" With Polar Coordinate Positional Embeddings.
			https://arxiv.org/abs/2509.10534
		[5] hunterFormsBS.bandSplit.lossComputation

		[6] ZFTurbo/Music-Source-Separation-Training
			https://github.com/ZFTurbo/Music-Source-Separation-Training
		[7] ZFTurbo/Music-Source-Separation-Training/train.py
			https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/train.py
		[8] torch.utils.checkpoint
			https://docs.pytorch.org/docs/stable/checkpoint.html
		"""
		device: torch.device = raw_audio.device
		x_is_mps: bool = device.type == 'mps'

		if raw_audio.ndim == 2:
			raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

		batch, channels, raw_audio_length = raw_audio.shape

		istft_length: int | None = raw_audio_length if self.match_input_audio_length else None

		if not ((not self.stereo and channels == 1) or (self.stereo and channels == 2)):
			message: str = f'I received `{channels = }` audio channel(s) but `{self.stereo = }`. I need `channels == 1` when `stereo` is `False`, or `channels == 2` when `stereo` is `True`.'
			raise ValueError(message)

		# to stft

		raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

		stft_window: Tensor = self.stft_window_fn(device=device)

		# RuntimeError: FFT operations are only supported on MacOS 14+  # noqa: ERA001
		# Since it's tedious to define whether we're on correct MacOS version - simple try-catch is used
		try:
			stft_repr: Tensor = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
		except RuntimeError:
			stft_repr = torch.stft(
				raw_audio.cpu() if x_is_mps else raw_audio,
				**self.stft_kwargs,
				window=stft_window.cpu() if x_is_mps else stft_window,
				return_complex=True,
			).to(device)
		stft_repr = torch.view_as_real(stft_repr)

		stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

		# merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
		stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

		# index out all frequencies for all frequency ranges across bands ascending in one go

		batch_arange: Tensor = torch.arange(batch, device=device)[..., None]

		# account for stereo

		x: Tensor = stft_repr[batch_arange, cast('Tensor', self.freq_indices)]

		# fold the complex (real and imag) into the frequencies dimension  # noqa: ERA001

		x = rearrange(x, 'b f t c -> b t (f c)')

		if self.use_torch_checkpoint:
			x = checkpoint(self.band_split, x, use_reentrant=False) # pyright: ignore[reportUnknownVariableType, reportAssignmentType]
		else:
			x = self.band_split(x)

		# axial / hierarchical attention

		store: list[Tensor | None] = [None] * len(self.layers)
		for i, transformer_block in enumerate(self.layers):
			layer: ModuleList = cast('ModuleList', transformer_block)
			time_transformer: Transformer = cast('Transformer', layer[-2])
			freq_transformer: Transformer = cast('Transformer', layer[-1])

			if self.skip_connection:
				# Sum all previous
				for j in range(i):
					x = x + raiseIfNone(store[j])

			x = rearrange(x, 'b t f d -> b f t d')
			x, ps = pack([x], '* t d')

			if self.use_torch_checkpoint:
				x = checkpoint(time_transformer, x, use_reentrant=False) # pyright: ignore[reportUnknownVariableType, reportAssignmentType]
			else:
				x = time_transformer(x)

			(x,) = unpack(x, ps, '* t d')
			x = rearrange(x, 'b f t d -> b t f d')
			x, ps = pack([x], '* f d')

			if self.use_torch_checkpoint:
				x = checkpoint(freq_transformer, x, use_reentrant=False) # pyright: ignore[reportUnknownVariableType, reportAssignmentType]
			else:
				x = freq_transformer(x)

			(x,) = unpack(x, ps, '* f d')

			if self.skip_connection:
				store[i] = x

		x = self.final_norm(x)

		if active_stem_ids is None:
			heads: ModuleList = self.mask_estimators
			stem_ids: list[int] = list(range(len(self.mask_estimators)))
		else:
			heads = ModuleList([self.mask_estimators[i] for i in active_stem_ids])
			stem_ids = active_stem_ids

		if self.use_torch_checkpoint:
			masks: Tensor = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in heads], dim=1) # pyright: ignore[reportArgumentType]
		else:
			masks = torch.stack([mask_estimator(x) for mask_estimator in heads], dim=1)
		masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)

		# modulate frequency representation

		stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

		# complex number multiplication

		stft_repr = torch.view_as_complex(stft_repr)
		masks = torch.view_as_complex(masks)
		masks = masks.type(stft_repr.dtype)

		# need to average the estimated mask for the overlapped frequencies

		scatter_indices: Tensor = repeat(cast('Tensor', self.freq_indices), 'f -> b n f t', b=batch, n=self.num_stems, t=stft_repr.shape[-1])

		stft_repr_expanded_stems: Tensor = repeat(stft_repr, 'b 1 ... -> b n ...', n=self.num_stems)
		masks_summed: Tensor = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

		denom: Tensor = repeat(cast('Tensor', self.num_bands_per_freq), 'f -> (f r) 1', r=channels)

		masks = masks_summed / denom.clamp(min=1e-8)

		# modulate stft repr with estimated mask

		stft_repr = stft_repr * masks

		# istft

		stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

		if self.zero_dc:
			stft_repr = stft_repr.index_fill(1, tensor(0, device=device), 0.0)

		try:
			recon_audio: Tensor = cast('Callable[..., Tensor]', torch.istft)(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=istft_length) # pyright: ignore[reportUnknownMemberType]
		except RuntimeError:
			recon_audio = cast('Callable[..., Tensor]', torch.istft)( # pyright: ignore[reportUnknownMemberType]
				stft_repr.cpu() if x_is_mps else stft_repr,
				**self.stft_kwargs,
				window=stft_window.cpu() if x_is_mps else stft_window,
				return_complex=False,
				length=istft_length,
			).to(device)

		recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', b=batch, s=self.audio_channels, n=self.num_stems)

		if not exists(target):
			return recon_audio

		return lossComputation(recon_audio=recon_audio, target=target, stem_ids=stem_ids, multi_stft=self.multi_stft, return_loss_breakdown=return_loss_breakdown)
