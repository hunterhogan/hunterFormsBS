# ruff: noqa: D101, D102, D103, E741
"""Use band-splitting, static mask-filter-bank, and training-loss utilities for source separation.

You can use this module to build the frequency-domain front end and the per-stem output heads shared
by BS-RoFormer [1] and Mel-Band RoFormer [2]. `BandSplit` projects grouped STFT frequency bins into a
common feature width. `MaskEstimator` converts band tokens back into complex subband masks. `MLP`
constructs the band-local affine blocks used inside `MaskEstimator`. `lossComputation` combines a
waveform-domain mean absolute error term with a multi-resolution complex-STFT term to produce the
training objective. `DEFAULT_FREQS_PER_BANDS` provides the standard non-overlapping frequency-bin
partition used by the BS-RoFormer front end. `mask_filter_bank_bs_roformer_default` and
`mask_filter_bank_mel_band_default` provide bundled static Boolean band-membership maps for the common
non-overlapping and mel-band layouts.

Contents
--------
Functions
	lossComputation
		Compute waveform MAE and multi-resolution complex-STFT MAE for selected stems and return one
		total loss or an expanded breakdown.
	MLP
		Build one feedforward projection sequence from an input width to an output width.

Classes
	BandSplit
		Project concatenated band slices to a shared feature width.
	MaskEstimator
		Estimate one concatenated subband mask from a stack of band tokens.

Variables
	DEFAULT_FREQS_PER_BANDS
		Standard non-overlapping frequency-bin partition for the BS-RoFormer front end.
	mask_filter_bank_bs_roformer_default
		Static Boolean band-membership map matching `DEFAULT_FREQS_PER_BANDS`.
	mask_filter_bank_mel_band_default
		Static Boolean band-membership map for the common 60-band mel layout at 44.1 kHz and
		`stft_n_fft=2048`.

References
----------
[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
	Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
[2] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
	Transcription https://arxiv.org/abs/2409.04702
"""
from __future__ import annotations

from einops import rearrange
from hunterFormsBS.theTypes import ParametersComputeLoss, ParametersSTFT
from more_itertools import loops
from torch import nn, Tensor, tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import default
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast, Literal, overload, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence

DEFAULT_FREQS_PER_BANDS: tuple[int, ...] = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)
mask_filter_bank_mel_band_default: Tensor = tensor(dtype=torch.bool, data=[[1]*7+[0]*1018,[0]*4+[1]*6+[0]*1015,[0]*7+[1]*6+[0]*1012,[0]*10+[1]*6+[0]*1009,[0]*13+[1]*6+[0]*1006,[0]*16+[1]*6+[0]*1003,[0]*19+[1]*6+[0]*1000,[0]*22+[1]*6+[0]*997,[0]*25+[1]*6+[0]*994,[0]*28+[1]*6+[0]*991,[0]*31+[1]*6+[0]*988,[0]*34+[1]*6+[0]*985,[0]*37+[1]*6+[0]*982,[0]*40+[1]*6+[0]*979,[0]*43+[1]*6+[0]*976,[0]*46+[1]*7+[0]*972,[0]*49+[1]*7+[0]*969,[0]*53+[1]*7+[0]*965,[0]*56+[1]*9+[0]*960,[0]*60+[1]*9+[0]*956,[0]*65+[1]*9+[0]*951,[0]*69+[1]*10+[0]*946,[0]*74+[1]*10+[0]*941,[0]*79+[1]*11+[0]*935,[0]*84+[1]*13+[0]*928,[0]*90+[1]*13+[0]*922,[0]*97+[1]*13+[0]*915,[0]*103+[1]*15+[0]*907,[0]*110+[1]*16+[0]*899,[0]*118+[1]*17+[0]*890,[0]*126+[1]*19+[0]*880,[0]*135+[1]*20+[0]*870,[0]*145+[1]*20+[0]*860,[0]*155+[1]*22+[0]*848,[0]*165+[1]*24+[0]*836,[0]*177+[1]*26+[0]*822,[0]*189+[1]*28+[0]*808,[0]*203+[1]*29+[0]*793,[0]*217+[1]*31+[0]*777,[0]*232+[1]*33+[0]*760,[0]*248+[1]*36+[0]*741,[0]*265+[1]*39+[0]*721,[0]*284+[1]*41+[0]*700,[0]*304+[1]*44+[0]*677,[0]*325+[1]*47+[0]*653,[0]*348+[1]*50+[0]*627,[0]*372+[1]*54+[0]*599,[0]*398+[1]*57+[0]*570,[0]*426+[1]*61+[0]*538,[0]*455+[1]*66+[0]*504,[0]*487+[1]*71+[0]*467,[0]*521+[1]*76+[0]*428,[0]*558+[1]*80+[0]*387,[0]*597+[1]*86+[0]*342,[0]*638+[1]*93+[0]*294,[0]*683+[1]*99+[0]*243,[0]*731+[1]*105+[0]*189,[0]*782+[1]*113+[0]*130,[0]*836+[1]*122+[0]*67,[0]*895+[1]*130])
"""Access the bundled common mel-band Boolean membership mask.

You can use `mask_filter_bank_mel_band_default` when the common `sample_rate=44100`,
`stft_n_fft=2048`, and `num_bands=60` layout should use one packaged static tensor.
"""
mask_filter_bank_bs_roformer_default: Tensor = tensor(dtype=torch.bool, data=[[1,1]+[0]*1023,[0,0,1,1]+[0]*1021,[0]*4+[1,1]+[0]*1019,[0]*6+[1,1]+[0]*1017,[0]*8+[1,1]+[0]*1015,[0]*10+[1,1]+[0]*1013,[0]*12+[1,1]+[0]*1011,[0]*14+[1,1]+[0]*1009,[0]*16+[1,1]+[0]*1007,[0]*18+[1,1]+[0]*1005,[0]*20+[1,1]+[0]*1003,[0]*22+[1,1]+[0]*1001,[0]*24+[1,1]+[0]*999,[0]*26+[1,1]+[0]*997,[0]*28+[1,1]+[0]*995,[0]*30+[1,1]+[0]*993,[0]*32+[1,1]+[0]*991,[0]*34+[1,1]+[0]*989,[0]*36+[1,1]+[0]*987,[0]*38+[1,1]+[0]*985,[0]*40+[1,1]+[0]*983,[0]*42+[1,1]+[0]*981,[0]*44+[1,1]+[0]*979,[0]*46+[1,1]+[0]*977,[0]*48+[1]*4+[0]*973,[0]*52+[1]*4+[0]*969,[0]*56+[1]*4+[0]*965,[0]*60+[1]*4+[0]*961,[0]*64+[1]*4+[0]*957,[0]*68+[1]*4+[0]*953,[0]*72+[1]*4+[0]*949,[0]*76+[1]*4+[0]*945,[0]*80+[1]*4+[0]*941,[0]*84+[1]*4+[0]*937,[0]*88+[1]*4+[0]*933,[0]*92+[1]*4+[0]*929,[0]*96+[1]*12+[0]*917,[0]*108+[1]*12+[0]*905,[0]*120+[1]*12+[0]*893,[0]*132+[1]*12+[0]*881,[0]*144+[1]*12+[0]*869,[0]*156+[1]*12+[0]*857,[0]*168+[1]*12+[0]*845,[0]*180+[1]*12+[0]*833,[0]*192+[1]*24+[0]*809,[0]*216+[1]*24+[0]*785,[0]*240+[1]*24+[0]*761,[0]*264+[1]*24+[0]*737,[0]*288+[1]*24+[0]*713,[0]*312+[1]*24+[0]*689,[0]*336+[1]*24+[0]*665,[0]*360+[1]*24+[0]*641,[0]*384+[1]*48+[0]*593,[0]*432+[1]*48+[0]*545,[0]*480+[1]*48+[0]*497,[0]*528+[1]*48+[0]*449,[0]*576+[1]*48+[0]*401,[0]*624+[1]*48+[0]*353,[0]*672+[1]*48+[0]*305,[0]*720+[1]*48+[0]*257,[0]*768+[1]*128+[0]*129,[0]*896+[1]*129])

class BandSplit(Module):
	"""Project band slices to a shared feature width.

	You can use this class to convert one concatenated band representation into one stack of band
	tokens with a common feature width. `BandSplit` treats the last axis of the input `Tensor` as
	consecutive band slices, applies one learned projection module to each slice, and returns one
	`Tensor` with an explicit band axis. `BandSplit` is the shared front-end projector used by
	`hunterFormsBS.bandSplitRotator.BandSplitRotator` [1], `hunterFormsBS.bs_roformer.BSRoformer` [2]
	and `hunterFormsBS.mel_band_roformer.MelBandRoformer` [3].

	Attributes
	----------
	dim_inputs : list[int]
		Width of each contiguous slice consumed from the last axis of the input `Tensor`.
	to_features : ModuleList
		Per-band projection module collection. Each entry transforms one slice width from `dim_inputs`
		to the shared feature width configured in `__init__`.

	See Also
	--------
	hunterFormsBS.bandSplitRotator.BandSplitRotator
		Unified separator that uses `BandSplit` as the band front end.
	hunterFormsBS.bs_roformer.BSRoformer
		Non-overlapping wrapper that uses `BandSplit` for the front-end projection.
	hunterFormsBS.mel_band_roformer.MelBandRoformer
		Overlapped mel-band wrapper that uses `BandSplit` for the front-end projection.

	References
	----------
	[1] hunterFormsBS.bandSplitRotator.BandSplitRotator

	[2] hunterFormsBS.bs_roformer.BSRoformer

	[3] hunterFormsBS.mel_band_roformer.MelBandRoformer
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

		outs: list[Tensor] = []
		for split_input, to_feature in zip(tuple_inputs, self.to_features, strict=True):
			split_output: Tensor = to_feature(split_input)
			outs.append(split_output)

		return torch.stack(outs, dim=-2)

@overload
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ParametersComputeLoss, *, return_loss_breakdown: Literal[True]) -> tuple[Tensor, tuple[Tensor, Tensor]]:...
@overload
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ParametersComputeLoss, *, return_loss_breakdown: Literal[False] = False) -> Tensor:...
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ParametersComputeLoss, *, return_loss_breakdown: bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
	"""Compute waveform and multi-resolution spectrogram loss for selected stems.

	You can use `lossComputation` to train one separator that reconstructs waveform audio and matches
	target spectrogram structure at multiple STFT window sizes. The loss family follows the
	BS-RoFormer objective [1] and the later Mel-RoFormer formulation family [2]. `lossComputation`
	selects the target stems indicated by `stem_ids`, trims `target` to the reconstructed sample
	count, computes one waveform-domain mean absolute error term, accumulates one complex-STFT mean
	absolute error term across the configured resolutions, and combines both terms into one total
	loss. `lossComputation` is the shared training-loss helper used by
	`hunterFormsBS.bandSplitRotator.BandSplitRotator` [3], `hunterFormsBS.bs_roformer.BSRoformer` [4],
	and `hunterFormsBS.mel_band_roformer.MelBandRoformer` [5].

	Parameters
	----------
	recon_audio : Tensor
		Reconstructed waveform estimate. `recon_audio` is expected to have shape `(batch, stem,
		audioChannel, time)` or a shape that is broadcast-compatible with `target[:, stem_ids]`.
	target : Tensor
		Reference waveform collection from which `stem_ids` selects the supervised stems. The last
		axis is interpreted as time. When `target.ndim == 2`, `lossComputation` inserts one singleton
		axis before the time axis so mono shorthand remains broadcast-compatible with mono
		reconstructions.
	stem_ids : list[int]
		Stem index list used to select the supervised subset from `target[:, stem_ids]`.
	multi_stft : ParametersComputeLoss
		Multi-resolution STFT configuration. `multi_stft` supplies `window_sizes`, `hop_length`,
		`n_fft`, `normalized`, `window_fn`, and `loss_weight`.
	return_loss_breakdown : bool = False
		When `True`, return both the combined loss and the two uncombined loss terms.

	Returns
	-------
	total_loss : Tensor
		Returned when `return_loss_breakdown` is `False`. `total_loss` equals the waveform-domain MAE
		plus `multi_stft['loss_weight']` times the accumulated multi-resolution complex-STFT MAE.
	loss_with_breakdown : tuple[Tensor, tuple[Tensor, Tensor]]
		Returned when `return_loss_breakdown` is `True`. The outer `Tensor` is `total_loss`. The inner
		pair contains `loss` and `multi_stft_resolution_loss`. `multi_stft['loss_weight']` is not
		applied to `multi_stft_resolution_loss` in this return value.

	Mathematics
	-----------
	multi-resolution loss : equation
	```
		Let y ≜ `target_sel`,  ŷ ≜ `recon_audio`,
			W ≜ `multi_stft['window_sizes']`,  α ≜ `multi_stft['loss_weight']`,
			Ψ ≜ complex-valued STFT operator,
			Ψ_w ≜ Ψ with length w windowing function

		ℓₜ = ‖ŷ − y‖₁
		ℓₛ = ∑_{w ∈ W} ‖Ψ_w(ŷ) − Ψ_w(y)‖₁
		ℒ = ℓₜ + α · ℓₛ

		where	ℓₜ ≜ `loss`,  ℓₛ ≜ `multi_stft_resolution_loss`,
				ℒ ≜ `total_loss`
	```

	PyTorch
	-------
	complex loss semantics : behavior
		`multi_stft_resolution_loss` is accumulated with `torch.nn.functional.l1_loss` [6] on the
		complex STFT tensors returned by `torch.stft` [7]. In PyTorch, `torch.nn.functional.l1_loss`
		is equivalent to `torch.abs(input - target).mean()` [6], so the implemented complex norm uses
		the complex modulus from `torch.abs` [8] rather than an explicit sum of separate real and
		imaginary MAEs.

	Sequence Alignment
	------------------
	target shortening : behavior
		`target` is trimmed to `target[..., : recon_audio.shape[-1]]` before stem selection. This
		prevents sample-count mismatch when callers reconstruct waveform audio with `torch.istft` and
		lose trailing samples [3].

	References
	----------
	[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
		Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
	[2] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
		Transcription https://arxiv.org/abs/2409.04702
	[3] hunterFormsBS.bandSplitRotator.BandSplitRotator

	[4] hunterFormsBS.bs_roformer.BSRoformer

	[5] hunterFormsBS.mel_band_roformer.MelBandRoformer

	[6] torch.nn.functional.l1_loss - PyTorch
		https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.l1_loss.html
	[7] torch.stft - PyTorch
		https://docs.pytorch.org/docs/stable/generated/torch.stft.html
	[8] torch.abs - PyTorch
		https://docs.pytorch.org/docs/stable/generated/torch.abs.html
	"""
	device: torch.device = recon_audio.device
	# if a target is passed in, calculate loss for learning
	if target.ndim == 2:
		target = rearrange(target, '... t -> ... 1 t')

	target = target[..., : recon_audio.shape[-1]]  # protect against lost length on istft

	target_sel: Tensor = target[:, stem_ids]
	loss: Tensor = F.l1_loss(recon_audio, target_sel)

	multi_stft_resolution_loss: Tensor = tensor(0.0, device=device)

	for window_size in multi_stft['window_sizes']:
		res_stft_kwargs = ParametersSTFT(
			hop_length=multi_stft['hop_length'],
			n_fft=max(window_size, multi_stft['n_fft']),
			normalized=multi_stft['normalized'],
			win_length=window_size,
		)

		recon_Y: Tensor = torch.stft(input=rearrange(recon_audio, 'b n s t -> (b n s) t'), return_complex=True, window=multi_stft['window_fn'](window_size, device=device), **res_stft_kwargs)
		target_Y: Tensor = torch.stft(input=rearrange(target_sel, 'b n s t -> (b n s) t'), return_complex=True, window=multi_stft['window_fn'](window_size, device=device), **res_stft_kwargs)

		multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

	weighted_multi_resolution_loss: Tensor = multi_stft_resolution_loss * multi_stft['loss_weight']

	total_loss: Tensor = loss + weighted_multi_resolution_loss

	if not return_loss_breakdown:
		return total_loss

	return total_loss, (loss, multi_stft_resolution_loss)

class MaskEstimator(Module):
	"""Estimate one concatenated subband mask from band tokens.

	You can use this class to convert one stack of band tokens into one concatenated band-local mask
	representation for a single separator head. `MaskEstimator` applies one learned projection head to
	each band on the penultimate axis of the input `Tensor`, then concatenates the band-local outputs
	along the last axis. The surrounding separator `hunterFormsBS.bandSplitRotator.BandSplitRotator`
	[1] later reshapes the concatenated output into real and imaginary mask values and, for
	overlapped band layouts, performs the overlap averaging step outside `MaskEstimator` [1][3].
	`MaskEstimator` corresponds to the multi-band mask-estimation family from BS-RoFormer [2] and the
	embedding-projection family from Mel-RoFormer [3]. Each band head uses
	`hunterFormsBS.bandSplit.MLP` [4] as the affine block before the final gate.

	Attributes
	----------
	dim_inputs : list[int]
		Width of each band-local output segment in the concatenated return value.
	to_freqs : ModuleList
		Per-band projection head collection. Each head maps one `dim`-wide band token to one
		band-local output segment whose width is given by `dim_inputs`.

	See Also
	--------
	hunterFormsBS.bandSplit.MLP
		Build the band-local affine block used before the final gate in each head.
	hunterFormsBS.bandSplitRotator.BandSplitRotator
		Use `MaskEstimator` as one separator head inside the full model.

	Implementation boundary
	-----------------------
	band-head operators : behavior
		`MaskEstimator` stores the band-local affine stack from `hunterFormsBS.bandSplit.MLP` [4] and
		the final gated output stage [6]. Any upstream normalization such as `RMSNorm` [5] must happen
		before `MaskEstimator` receives band token `Tensor` `x`.
	package default head template : behavior
		When `depth = 1`, `mlp_expansion_factor = 4`, and `activation` is the hyperbolic tangent
		activation, `MaskEstimator` matches the default hidden-width and activation pattern used by the
		published model family [2][3][6].

	References
	----------
	[1] hunterFormsBS.bandSplitRotator.BandSplitRotator

	[2] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation
		with Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
	[3] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
		Transcription https://arxiv.org/abs/2409.04702

	[4] hunterFormsBS.bandSplit.MLP

	[5] Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
		https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
	[6] Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language Modeling with
		Gated Convolutional Networks. https://proceedings.mlr.press/v70/dauphin17a.html
	"""
	def __init__(
		self,
		dim: int,
		dim_inputs: Sequence[int],
		*,
		depth: int,
		mlp_expansion_factor: int = 4,
		activation: type[nn.Module] = nn.Tanh,
		segm_out_bins: int | None = None,
		segm_out_channels: int = 4,
		segm_base_channels: int = 64,
		segm_base_depth: int = 2,
		segm_num_hyperedges: int = 32,
		segm_num_heads: int = 8,
		segm_backbone_channels: tuple[int, int, int, int, int] | None = None,
		segm_hyperace_k: int = 2,
		segm_hyperace_l: int = 1,
		segm_hyperace_c_h: float = 0.5,
		segm_hyperace_c_l: float = 0.25,
		segm_hyperace_c3ah_expansion: float = 1.0,
		segm_hyperace_low_order_depth: int = 1,
		segm_hyperace_low_order_kernel: int = 3,
		segm_hyperace_low_order_expansion: float = 1.0,
		segm_hyperace_out_channels: int | None = None,
		segm_decoder_channels: list[int] | tuple[int, int, int, int] | None = None,
		segm_decoder_block_depth: int = 1,
		segm_decoder_block_kernel: int = 3,
		segm_decoder_block_expansion: float = 0.5,
		segm_upsample_scales: tuple[int, int, int, int] = (2, 2, 2, 2),
		segm_upsample_tfc_tdf_depth: int = 2,
		segm_upsample_tfc_tdf_bn: int = 4,
		segm_activation: type[nn.Module] = nn.SiLU,
		segm_norm_eps: float = 1e-8,
		segm_norm_affine: bool = True,
		segm_conv_bias: bool = False,
		segm_linear_bias: bool = False,
		use_hyperACE: bool = False,
	) -> None:
		"""Configure one mask-projection head per band.

		You can use `__init__` to specify how many band heads `MaskEstimator` builds, how wide each
		band-local output segment should be, and how deep the shared head template should be.
		`__init__` stores the per-band output widths in `self.dim_inputs` and creates one band-local
		projection head for each band.

		Parameters
		----------
		dim : int
			Feature width of each input band token.
		dim_inputs : Sequence[int]
			Output width sequence for the band-local segments. Entry `dim_inputs[k]` becomes the
			last-axis width emitted by head index `k`.
		depth : int
			Number of hidden-width repeats inside each band-local `MLP`. `depth = 1` yields the
			two-linear-layer pattern used by the paper-default head.
		mlp_expansion_factor : int = 4
			Hidden-width multiplier relative to `dim`. The hidden width becomes `dim *
			mlp_expansion_factor`. The default value `4` matches the hidden-width ratio reported for
			the mask-estimation head.
		activation : type[nn.Module] = nn.Tanh
			Activation class instantiated between non-final linear layers inside each band-local
			`MLP`. The default value corresponds to the hyperbolic tangent activation used by the
			paper-default head.
		"""
		super().__init__()
		self.dim_inputs: list[int] = list(dim_inputs)
		self.to_freqs: ModuleList = ModuleList([])
		dim_hidden: int = dim * mlp_expansion_factor
		self.use_hyperACE: bool = use_hyperACE

		for dim_in in self.dim_inputs:
			mlp: nn.Sequential = nn.Sequential(MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth, activation=activation), nn.GLU(dim=-1))
			self.to_freqs.append(mlp)

		if self.use_hyperACE:
			# https://huggingface.co/pcunwa/BS-Roformer-HyperACE/commit/76d35e479166ceb7dc07a457a37760de4f7091c1
			segm_out_bins = sum(dim_inputs) // segm_out_channels if segm_out_bins is None else segm_out_bins
			self.segm = SegmModel(
				in_bands=len(dim_inputs),
				in_dim=dim,
				out_bins=segm_out_bins,
				out_channels=segm_out_channels,
				base_channels=segm_base_channels,
				base_depth=segm_base_depth,
				num_hyperedges=segm_num_hyperedges,
				num_heads=segm_num_heads,
				backbone_channels=segm_backbone_channels,
				hyperace_k=segm_hyperace_k,
				hyperace_l=segm_hyperace_l,
				hyperace_c_h=segm_hyperace_c_h,
				hyperace_c_l=segm_hyperace_c_l,
				hyperace_c3ah_expansion=segm_hyperace_c3ah_expansion,
				hyperace_low_order_depth=segm_hyperace_low_order_depth,
				hyperace_low_order_kernel=segm_hyperace_low_order_kernel,
				hyperace_low_order_expansion=segm_hyperace_low_order_expansion,
				hyperace_out_channels=segm_hyperace_out_channels,
				decoder_channels=segm_decoder_channels,
				decoder_block_depth=segm_decoder_block_depth,
				decoder_block_kernel=segm_decoder_block_kernel,
				decoder_block_expansion=segm_decoder_block_expansion,
				upsample_scales=segm_upsample_scales,
				upsample_tfc_tdf_depth=segm_upsample_tfc_tdf_depth,
				upsample_tfc_tdf_bn=segm_upsample_tfc_tdf_bn,
				activation=segm_activation,
				norm_eps=segm_norm_eps,
				norm_affine=segm_norm_affine,
				conv_bias=segm_conv_bias,
				linear_bias=segm_linear_bias,
			)

	def forward(self, x: Tensor) -> Tensor:
		"""Estimate one concatenated mask from band features `x`.

		You can use `forward` when `x` contains one learned token per band on the penultimate axis.
		`forward` sends each band token to the matching projection head in `self.to_freqs`, then
		concatenates the band-local outputs along the last axis. The return value keeps band order but
		removes the explicit band axis, so callers can reinterpret contiguous output segments as one
		complex mask layout for later reshaping. `forward` does not average overlapping frequency bins;
		later package code performs any overlap-aware reconstruction step after reshaping.

		Parameters
		----------
		x : Tensor
			Band-token tensor. `x.shape[-2]` must equal `len(self.dim_inputs)`. In current callers,
			`x` usually has shape `(batch, time, band, feature)`.

		Returns
		-------
		concatenated_mask : Tensor
			Concatenated band-local output tensor. If the leading shape of `x` is `leadingShape`, the
			return value has shape `(*leadingShape, sum(self.dim_inputs))`.

		PyTorch Tensor Shapes
		---------------------
		input band axis : relation
			`forward` interprets `x[..., k, :]` as the feature tensor for band index `k`.
		per-band output width : relation
			Head index `k` maps `x[..., k, :]` to one tensor whose last-axis width is
			`self.dim_inputs[k]`.
		output concatenation : relation
			`forward` concatenates the projected band outputs in ascending band order along the last
			axis.

		Mathematics
		-----------
		band-head family : equation
		```
			Let X ≜ `x`,  K ≜ len(`self.dim_inputs`),
				zₖ ≜ `self.dim_inputs[k]`,  xₖ ≜ X[..., k, :]

			Φₖ : ℝ^{…, d} → ℝ^{…, zₖ}    ∀ k ∈ {0, …, K − 1}
			yₖ = Φₖ(xₖ)                  ∀ k ∈ {0, …, K − 1}
			Z = ∑ₖ zₖ
			y = y₀ ‖ y₁ ‖ ⋯ ‖ y_{K−1} ∈ ℝ^{…, Z}

			where  y ≜ `concatenated_mask`
		```
		default head specialization : equation
		```
			Let  L ≜ `depth`,  γ ≜ `mlp_expansion_factor`,  φ ≜ `activation`

			L = 1
			γ = 4
			φ = tanh

			Wₖ,1 : ℝ^{d} → ℝ^{4d}
			Wₖ,2 : ℝ^{4d} → ℝ^{2zₖ}
			hₖ = φ(Wₖ,1 xₖ + bₖ,1)
			[aₖ ‖ gₖ] = Wₖ,2 hₖ + bₖ,2
			yₖ = aₖ ⊙ σ(gₖ)
		```

		Separator integration
		---------------------
		non-overlapping mask assembly : behavior
			In non-overlapping band layouts, the concatenated output can be reinterpreted as one full
			complex mask by frequency-axis concatenation [1].
		overlap handling outside `MaskEstimator` : behavior
			In overlapped mel-band layouts, `hunterFormsBS.bandSplitRotator.BandSplitRotator` [3]
			performs the later overlap averaging step after reshaping the concatenated output back to
			band-local mask slices [2][3].
		normalization boundary : behavior
			`forward` does not apply per-band `RMSNorm` [4]. `forward` only invokes the stored
			band-local affine block and final gate [5].

		References
		----------
		[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation
			with Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
		[2] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
			Transcription https://arxiv.org/abs/2409.04702
		[3] hunterFormsBS.bandSplitRotator.BandSplitRotator

		[4] Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
			https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
		[5] Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language Modeling with
			Gated Convolutional Networks. https://proceedings.mlr.press/v70/dauphin17a.html
		"""
		if self.use_hyperACE:
			y: Tensor = rearrange(x, 'b t f c -> b c t f')
			y = self.segm(y)
			y = rearrange(y, 'b c t f -> b t (f c)')

		tensor_unbound: tuple[Tensor, ...] = x.unbind(dim=-2)

		outs: list[Tensor] = []

		for band_features, mlp in zip(tensor_unbound, self.to_freqs, strict=True):
			freq_out: Tensor = mlp(band_features)
			outs.append(freq_out)

		out: Tensor = torch.cat(outs, dim=-1)

		if self.use_hyperACE:
			out = out + y # pyright: ignore[reportPossiblyUnboundVariable]

		return out

def MLP(dim_in: int, dim_out: int, dim_hidden: int | None = None, depth: int = 1, activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
	"""Build one feedforward projection from `dim_in` to `dim_out`.

	You can use `MLP` to create one `nn.Sequential` made of linear layers with one intermediate
	activation after every non-final linear layer. `MLP` does not apply normalization or output gating
	by itself. `hunterFormsBS.bandSplit.MaskEstimator` [1] uses `MLP` as the band-local affine block
	before the final gate in the BS-RoFormer [2] and Mel-RoFormer [3] mask heads.

	Parameters
	----------
	dim_in : int
		Input feature width for the first linear layer.
	dim_out : int
		Output feature width for the last linear layer.
	dim_hidden : int | None = None
		Hidden feature width for repeated intermediate linear layers. When `dim_hidden` is `None`,
		`MLP` uses `dim_in`.
	depth : int = 1
		Number of repeated `dim_hidden` stages between the input layer and the output layer. `depth =
		0` yields a single linear layer. `depth = 1` yields one hidden layer.
	activation : type[nn.Module] = nn.Tanh
		Activation class instantiated after each non-final linear layer.

	Returns
	-------
	network : nn.Sequential
		Sequential stack of alternating linear layers and instantiated `activation` modules. The last
		module is always linear.

	Mathematics
	-----------
	linear stack : equation
	```
		Let x denote the input vector,  L ≜ `depth`,
			d₀ ≜ `dim_in`,  d_out ≜ `dim_out`,  d_h denote the effective hidden width

		d₁ = ⋯ = d_L = d_h
		h₀ = x
		hᵢ₊₁ = φ(Wᵢ hᵢ + bᵢ)    ∀ i ∈ {0, …, L − 1}
		y = W_L h_L + b_L

		where  d_h ≜ hidden width determined from `dim_hidden`,  φ ≜ `activation`
	```
	default mask-head affine block : equation
	```
		Let  x denote one band-local input vector,  d ≜ `dim_in`

		L = 1
		d_h = 4d
		φ = tanh
		h = φ(W₁ x + b₁)
		g = W₂ h + b₂

		where  g ≜ affine output returned by `MLP` before the later gate in
			`hunterFormsBS.bandSplit.MaskEstimator`
	```

	Layer Construction
	------------------
	activation instantiation : behavior
		`MLP` instantiates a fresh `activation()` module after each non-final linear layer. `MLP` does
		not append an activation after the last linear layer.

	See Also
	--------
	hunterFormsBS.bandSplit.MaskEstimator
		Use `MLP` as the band-local affine block before the final gate.

	References
	----------
	[1] hunterFormsBS.bandSplit.MaskEstimator

	[2] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation
		with Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
	[3] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
		Transcription https://arxiv.org/abs/2409.04702
	"""
	dim_hidden = default(dim_hidden, dim_in)

	net: list[nn.Module] = []
	dims: tuple[int, ...] = (dim_in, *((dim_hidden,) * depth), dim_out)

	for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):  # noqa: RUF007
		is_last: bool = ind == (len(dims) - 2)

		net.append(nn.Linear(layer_dim_in, layer_dim_out))

		if is_last:
			continue

		net.append(activation())

	return nn.Sequential(*net)

class Conv(nn.Module):
	def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, act: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bias)
		self.bn = nn.InstanceNorm2d(c2, affine=norm_affine, eps=norm_eps)
		self.act: nn.Module = activation() if act else nn.Identity()

	def forward(self, x: Tensor) -> Tensor:
		return self.act(self.bn(self.conv(x)))

@overload
def autopad(k: int, p: int | None = None) -> int:...
@overload
def autopad(k: list[int], p: list[int] | None = None) -> list[int]:...
def autopad(k: int | list[int], p: int | list[int] | None = None) -> int | list[int]:
	if p is None:
		p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
	return p

class DSConv(nn.Module):
	def __init__(self, c1: int, c2: int, k: int = 3, s: int | tuple[int, int] = 1, p: int | None = None
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, act: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=bias)
		self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=bias)
		self.bn = nn.InstanceNorm2d(c2, affine=norm_affine, eps=norm_eps)
		self.act: nn.Module = activation() if act else nn.Identity()

	def forward(self, x: Tensor) -> Tensor:
		return self.act(self.bn(self.pwconv(self.dwconv(x))))

class DS_Bottleneck(nn.Module):
	def __init__(self, c1: int, c2: int, k: int = 3
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, shortcut: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		c_: int = c1
		self.dsconv1: DSConv = DSConv(c1, c_, k=3, s=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.dsconv2: DSConv = DSConv(c_, c2, k=k, s=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.shortcut: bool = shortcut and c1 == c2

	def forward(self, x: Tensor) -> Tensor:
		return x + self.dsconv2(self.dsconv1(x)) if self.shortcut else self.dsconv2(self.dsconv1(x))

class DS_C3k(nn.Module):
	def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5, activation: type[nn.Module] = nn.SiLU, norm_eps: float = 1e-8, *, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		c_ = int(c2 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv2: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv3: Conv = Conv(2 * c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.m = nn.Sequential(
			*[
				DS_Bottleneck(c_, c_, k=k, shortcut=True, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
				for _ in range(n)
			]
		)

	def forward(self, x: Tensor) -> Tensor:
		return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class DS_C3k2(nn.Module):
	def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5, activation: type[nn.Module] = nn.SiLU, norm_eps: float = 1e-8, *, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		c_ = int(c2 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.m: DS_C3k = DS_C3k(c_, c_, n=n, k=k, e=1.0, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv2: Conv = Conv(c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)

	def forward(self, x: Tensor) -> Tensor:
		x_ = self.cv1(x)
		x_ = self.m(x_)
		return self.cv2(x_)

class AdaptiveHyperedgeGeneration(nn.Module):
	def __init__(self, in_channels: int, num_hyperedges: int, num_heads: int = 8, *, linear_bias: bool = False) -> None:
		super().__init__()
		self.num_hyperedges: int = num_hyperedges
		self.num_heads: int = num_heads
		self.head_dim: int = in_channels // num_heads

		self.global_proto: nn.Parameter = nn.Parameter(torch.randn(num_hyperedges, in_channels))

		self.context_mapper: nn.Linear = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=linear_bias)

		self.query_proj: nn.Linear = nn.Linear(in_channels, in_channels, bias=linear_bias)

		self.scale: float = self.head_dim**-0.5

	def forward(self, x: Tensor) -> Tensor:
		B, N, C = x.shape

		f_avg: Tensor = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
		f_max: Tensor = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
		f_ctx: Tensor = torch.cat((f_avg, f_max), dim=1)

		delta_P = self.context_mapper(f_ctx).view(B, self.num_hyperedges, C)
		P = self.global_proto.unsqueeze(0) + delta_P

		z = self.query_proj(x)

		z = z.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

		P = P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

		sim = (z @ P) * self.scale

		s_bar = sim.mean(dim=1)

		A: Tensor = F.softmax(s_bar.permute(0, 2, 1), dim=-1)

		return A

class HypergraphConvolution(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, activation: type[nn.Module] = nn.SiLU, *, linear_bias: bool = False) -> None:
		super().__init__()
		self.W_e = nn.Linear(in_channels, in_channels, bias=linear_bias)
		self.W_v = nn.Linear(in_channels, out_channels, bias=linear_bias)
		self.act: nn.Module = activation()

	def forward(self, x: Tensor, A: Tensor) -> Tensor:
		f_m: Tensor = torch.bmm(A, x)
		f_m = self.act(self.W_e(f_m))

		x_out: Tensor = torch.bmm(A.transpose(1, 2), f_m)
		x_out = self.act(self.W_v(x_out))

		return x + x_out

class AdaptiveHypergraphComputation(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, num_hyperedges: int = 8, num_heads: int = 8, activation: type[nn.Module] = nn.SiLU, *, linear_bias: bool = False) -> None:
		super().__init__()
		self.adaptive_hyperedge_gen: AdaptiveHyperedgeGeneration = AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads, linear_bias=linear_bias)
		self.hypergraph_conv: HypergraphConvolution = HypergraphConvolution(in_channels, out_channels, activation=activation, linear_bias=linear_bias)

	def forward(self, x: Tensor) -> Tensor:
		B, _C, H, W = x.shape
		x_flat: Tensor = x.flatten(2).permute(0, 2, 1)

		A = self.adaptive_hyperedge_gen(x_flat)

		x_out_flat = self.hypergraph_conv(x_flat, A)

		return x_out_flat.permute(0, 2, 1).view(B, -1, H, W)

class C3AH(nn.Module):
	def __init__(
		self,
		c1: int,
		c2: int,
		num_hyperedges: int = 8,
		num_heads: int = 8,
		e: float = 0.5,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()
		c_ = int(c1 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.cv2: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.ahc: AdaptiveHypergraphComputation = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads, activation=activation, linear_bias=linear_bias)
		self.cv3: Conv = Conv(2 * c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)

	def forward(self, x: Tensor) -> Tensor:
		x_lateral = self.cv1(x)
		x_ahc = self.ahc(self.cv2(x))
		return self.cv3(torch.cat((x_ahc, x_lateral), dim=1))

class HyperACE(nn.Module):
	def __init__(
		self,
		in_channels: list[int],
		out_channels: int,
		num_hyperedges: int = 8,
		num_heads: int = 8,
		k: int = 2,
		l: int = 1,
		c_h: float = 0.5,
		c_l: float = 0.25,
		c3ah_expansion: float = 1.0,
		low_order_depth: int = 1,
		low_order_kernel: int = 3,
		low_order_expansion: float = 1.0,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()

		c2, c3, c4, c5 = in_channels
		c_mid: int = c4

		self.fuse_conv: Conv = Conv(
			c2 + c3 + c4 + c5, c_mid, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)

		self.c_h = int(c_mid * c_h)
		self.c_l = int(c_mid * c_l)
		self.c_s: int = c_mid - self.c_h - self.c_l
		if self.c_s <= 0:
			message: str = (
				f"I computed `{self.c_s = }`, indicative of a channel split problem, from `{c_mid = }`, `{c_h = }`, `{c_l = }`, "
				f"`{self.c_h = }`, and `{self.c_l = }`, but I need `self.c_s` to be greater than 0."
			)
			raise ValueError(message)

		self.high_order_branch = nn.ModuleList(
			[
				C3AH(
					self.c_h,
					self.c_h,
					num_hyperedges,
					num_heads,
					e=c3ah_expansion,
					activation=activation,
					norm_eps=norm_eps,
					norm_affine=norm_affine,
					conv_bias=conv_bias,
					linear_bias=linear_bias,
				)
				for _index in loops(k)
			]
		)
		self.high_order_fuse: Conv = Conv(
			self.c_h * k, self.c_h, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)

		self.low_order_branch = nn.Sequential(
			*[
				DS_C3k(
					self.c_l,
					self.c_l,
					n=low_order_depth,
					k=low_order_kernel,
					e=low_order_expansion,
					activation=activation,
					norm_eps=norm_eps,
					norm_affine=norm_affine,
					bias=conv_bias,
				)
				for _ in range(l)
			]
		)

		self.final_fuse: Conv = Conv(
			self.c_h + self.c_l + self.c_s,
			out_channels,
			1,
			1,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)

	def forward(self, x: list[Tensor]) -> Tensor:
		B2, B3, B4, B5 = x

		_B, _C, H4, W4 = B4.shape

		B2_resized: Tensor = F.interpolate(B2, size=(H4, W4), mode='bilinear', align_corners=False)
		B3_resized: Tensor = F.interpolate(B3, size=(H4, W4), mode='bilinear', align_corners=False)
		B5_resized: Tensor = F.interpolate(B5, size=(H4, W4), mode='bilinear', align_corners=False)

		x_b = self.fuse_conv(torch.cat((B2_resized, B3_resized, B4, B5_resized), dim=1))

		x_h, x_l, x_s = torch.split(x_b, [self.c_h, self.c_l, self.c_s], dim=1)

		x_h_outs: list[Tensor] = [m(x_h) for m in self.high_order_branch]
		x_h_fused = self.high_order_fuse(torch.cat(x_h_outs, dim=1))

		x_l_out = self.low_order_branch(x_l)

		return self.final_fuse(torch.cat((x_h_fused, x_l_out, x_s), dim=1))

class GatedFusion(nn.Module):
	def __init__(self, in_channels: int) -> None:
		super().__init__()
		self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

	def forward(self, f_in: Tensor, h: Tensor) -> Tensor:
		if f_in.shape[1] != h.shape[1]:
			message: str = (
				f"I received `{f_in.shape = }` and `{h.shape = }`, but I need the number of channels to match, so "
				f"`{f_in.shape[1] = }` to equal `{h.shape[1] = }`."
			)
			raise ValueError(message)
		return f_in + self.gamma * h

class Backbone(nn.Module):
	def __init__(
		self,
		in_channels: int = 256,
		base_channels: int = 64,
		base_depth: int = 3,
		channels: tuple[int, int, int, int, int] | None = None,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
	) -> None:
		super().__init__()
		if channels is None:
			c2: int = base_channels
			c3 = 256
			c4 = 384
			c5 = 512
			c6 = 768
		else:
			c2, c3, c4, c5, c6 = channels

		self.stem: DSConv = DSConv(
			in_channels, c2, k=3, s=(2, 1), p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)

		self.p2 = nn.Sequential(
			DSConv(c2, c3, k=3, s=(2, 1), p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
			DS_C3k2(c3, c3, n=base_depth, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
		)

		self.p3 = nn.Sequential(
			DSConv(c3, c4, k=3, s=(2, 1), p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
			DS_C3k2(c4, c4, n=base_depth * 2, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
		)

		self.p4 = nn.Sequential(
			DSConv(c4, c5, k=3, s=2, p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
			DS_C3k2(c5, c5, n=base_depth * 2, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
		)

		self.p5 = nn.Sequential(
			DSConv(c5, c6, k=3, s=2, p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
			DS_C3k2(c6, c6, n=base_depth, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
		)

		self.out_channels: list[int] = [c3, c4, c5, c6]

	def forward(self, x: Tensor) -> list[Tensor]:
		x = self.stem(x)
		x2 = self.p2(x)
		x3 = self.p3(x2)
		x4 = self.p4(x3)
		x5 = self.p5(x4)
		return [x2, x3, x4, x5]

class Decoder(nn.Module):
	def __init__(
		self,
		encoder_channels: list[int],
		hyperace_out_c: int,
		decoder_channels: list[int],
		block_depth: int = 1,
		block_kernel: int = 3,
		block_expansion: float = 0.5,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
	) -> None:
		super().__init__()
		c_p2, c_p3, c_p4, c_p5 = encoder_channels
		c_d2, c_d3, c_d4, c_d5 = decoder_channels

		self.h_to_d5: Conv = Conv(hyperace_out_c, c_d5, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.h_to_d4: Conv = Conv(hyperace_out_c, c_d4, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.h_to_d3: Conv = Conv(hyperace_out_c, c_d3, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.h_to_d2: Conv = Conv(hyperace_out_c, c_d2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)

		self.fusion_d5: GatedFusion = GatedFusion(c_d5)
		self.fusion_d4: GatedFusion = GatedFusion(c_d4)
		self.fusion_d3: GatedFusion = GatedFusion(c_d3)
		self.fusion_d2: GatedFusion = GatedFusion(c_d2)

		self.skip_p5: Conv = Conv(c_p5, c_d5, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.skip_p4: Conv = Conv(c_p4, c_d4, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.skip_p3: Conv = Conv(c_p3, c_d3, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.skip_p2: Conv = Conv(c_p2, c_d2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)

		self.up_d5: DS_C3k2 = DS_C3k2(
			c_d5,
			c_d4,
			n=block_depth,
			k=block_kernel,
			e=block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)
		self.up_d4: DS_C3k2 = DS_C3k2(
			c_d4,
			c_d3,
			n=block_depth,
			k=block_kernel,
			e=block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)
		self.up_d3: DS_C3k2 = DS_C3k2(
			c_d3,
			c_d2,
			n=block_depth,
			k=block_kernel,
			e=block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)

		self.final_d2: DS_C3k2 = DS_C3k2(
			c_d2,
			c_d2,
			n=block_depth,
			k=block_kernel,
			e=block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)

	def forward(self, enc_feats: list[Tensor], h_ace: Tensor) -> Tensor:
		p2, p3, p4, p5 = enc_feats

		d5 = self.skip_p5(p5)
		h_d5 = self.h_to_d5(F.interpolate(h_ace, size=d5.shape[2:], mode='bilinear'))
		d5 = self.fusion_d5(d5, h_d5)

		d5_up: Tensor = F.interpolate(d5, size=p4.shape[2:], mode='bilinear')
		d4_skip = self.skip_p4(p4)
		d4 = self.up_d5(d5_up) + d4_skip

		h_d4 = self.h_to_d4(F.interpolate(h_ace, size=d4.shape[2:], mode='bilinear'))
		d4 = self.fusion_d4(d4, h_d4)

		d4_up: Tensor = F.interpolate(d4, size=p3.shape[2:], mode='bilinear')
		d3_skip = self.skip_p3(p3)
		d3 = self.up_d4(d4_up) + d3_skip

		h_d3 = self.h_to_d3(F.interpolate(h_ace, size=d3.shape[2:], mode='bilinear'))
		d3 = self.fusion_d3(d3, h_d3)

		d3_up: Tensor = F.interpolate(d3, size=p2.shape[2:], mode='bilinear')
		d2_skip = self.skip_p2(p2)
		d2 = self.up_d3(d3_up) + d2_skip

		h_d2 = self.h_to_d2(F.interpolate(h_ace, size=d2.shape[2:], mode='bilinear'))
		d2 = self.fusion_d2(d2, h_d2)

		return self.final_d2(d2)

class TFC_TDF(nn.Module):
	def __init__(self, in_c: int, c: int, l: int, f: int, bn: int = 4, activation: type[nn.Module] = nn.SiLU, norm_eps: float = 1e-8, *, norm_affine: bool = True, conv_bias: bool = False, linear_bias: bool = False) -> None:
		super().__init__()

		self.blocks = nn.ModuleList()
		for _index in loops(l):
			block = nn.Module()

			block.tfc1 = nn.Sequential(
				nn.InstanceNorm2d(in_c, affine=norm_affine, eps=norm_eps), activation(), nn.Conv2d(in_c, c, 3, 1, 1, bias=conv_bias)
			)
			block.tdf = nn.Sequential(
				nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps),
				activation(),
				nn.Linear(f, f // bn, bias=linear_bias),
				nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps),
				activation(),
				nn.Linear(f // bn, f, bias=linear_bias),
			)
			block.tfc2 = nn.Sequential(
				nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps), activation(), nn.Conv2d(c, c, 3, 1, 1, bias=conv_bias)
			)
			block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=conv_bias)

			self.blocks.append(block)
			in_c = c

	def forward(self, x: Tensor) -> Tensor:
		for block in self.blocks:
			s: Tensor = cast('Callable[[Tensor], Tensor]', block.shortcut)(x)
			x = cast('Callable[[Tensor], Tensor]', block.tfc1)(x)
			x = x + cast('Callable[[Tensor], Tensor]', block.tdf)(x)
			x = cast('Callable[[Tensor], Tensor]', block.tfc2)(x)
			x = x + s
		return x

class FreqPixelShuffle(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		scale: int,
		f: int,
		tfc_tdf_depth: int = 2,
		tfc_tdf_bn: int = 4,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()
		self.scale: int = scale
		self.conv: DSConv = DSConv(
			in_channels, out_channels * scale, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)
		self.out_conv: TFC_TDF = TFC_TDF(
			out_channels,
			out_channels,
			tfc_tdf_depth,
			f,
			bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.conv(x)
		B, C_r, H, W = x.shape
		out_c: int = C_r // self.scale

		x = x.view(B, out_c, self.scale, H, W)

		x = x.permute(0, 1, 3, 4, 2).contiguous()
		x = x.view(B, out_c, H, W * self.scale)

		return self.out_conv(x)

class ProgressiveUpsampleHead(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		target_bins: int = 1025,
		in_bands: int = 62,
		upsample_scales: tuple[int, int, int, int] = (2, 2, 2, 2),
		tfc_tdf_depth: int = 2,
		tfc_tdf_bn: int = 4,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()
		self.target_bins: int = target_bins

		c: int = in_channels
		scale1, scale2, scale3, scale4 = upsample_scales
		f1: int = in_bands * scale1
		f2: int = f1 * scale2
		f3: int = f2 * scale3
		f4: int = f3 * scale4

		self.block1: FreqPixelShuffle = FreqPixelShuffle(
			c,
			c // 2,
			scale=scale1,
			f=f1,
			tfc_tdf_depth=tfc_tdf_depth,
			tfc_tdf_bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)
		self.block2: FreqPixelShuffle = FreqPixelShuffle(
			c // 2,
			c // 4,
			scale=scale2,
			f=f2,
			tfc_tdf_depth=tfc_tdf_depth,
			tfc_tdf_bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)
		self.block3: FreqPixelShuffle = FreqPixelShuffle(
			c // 4,
			c // 8,
			scale=scale3,
			f=f3,
			tfc_tdf_depth=tfc_tdf_depth,
			tfc_tdf_bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)
		self.block4: FreqPixelShuffle = FreqPixelShuffle(
			c // 8,
			c // 16,
			scale=scale4,
			f=f4,
			tfc_tdf_depth=tfc_tdf_depth,
			tfc_tdf_bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)

		self.final_conv = nn.Conv2d(c // 16, out_channels, kernel_size=3, stride=1, padding='same', bias=conv_bias)

	def forward(self, x: Tensor) -> Tensor:

		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		if x.shape[-1] != self.target_bins:
			x = F.interpolate(x, size=(x.shape[2], self.target_bins), mode='bilinear', align_corners=False)

		return self.final_conv(x)

class SegmModel(nn.Module):
	def __init__(
		self,
		in_bands: int = 62,
		in_dim: int = 256,
		out_bins: int = 1025,
		out_channels: int = 4,
		base_channels: int = 64,
		base_depth: int = 2,
		num_hyperedges: int = 32,
		num_heads: int = 8,
		backbone_channels: tuple[int, int, int, int, int] | None = None,
		hyperace_k: int = 2,
		hyperace_l: int = 1,
		hyperace_c_h: float = 0.5,
		hyperace_c_l: float = 0.25,
		hyperace_c3ah_expansion: float = 1.0,
		hyperace_low_order_depth: int = 1,
		hyperace_low_order_kernel: int = 3,
		hyperace_low_order_expansion: float = 1.0,
		hyperace_out_channels: int | None = None,
		decoder_channels: list[int] | tuple[int, int, int, int] | None = None,
		decoder_block_depth: int = 1,
		decoder_block_kernel: int = 3,
		decoder_block_expansion: float = 0.5,
		upsample_scales: tuple[int, int, int, int] = (2, 2, 2, 2),
		upsample_tfc_tdf_depth: int = 2,
		upsample_tfc_tdf_bn: int = 4,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()

		self.backbone = Backbone(
			in_channels=in_dim,
			base_channels=base_channels,
			base_depth=base_depth,
			channels=backbone_channels,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
		)
		enc_channels: list[int] = self.backbone.out_channels
		c2, c3, c4, c5 = enc_channels

		hyperace_in_channels: list[int] = enc_channels
		hyperace_out_channels = c4 if hyperace_out_channels is None else hyperace_out_channels
		self.hyperace = HyperACE(
			hyperace_in_channels,
			hyperace_out_channels,
			num_hyperedges,
			num_heads,
			k=hyperace_k,
			l=hyperace_l,
			c_h=hyperace_c_h,
			c_l=hyperace_c_l,
			c3ah_expansion=hyperace_c3ah_expansion,
			low_order_depth=hyperace_low_order_depth,
			low_order_kernel=hyperace_low_order_kernel,
			low_order_expansion=hyperace_low_order_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)

		decoder_channels = [c2, c3, c4, c5] if decoder_channels is None else list(decoder_channels)
		self.decoder = Decoder(
			enc_channels,
			hyperace_out_channels,
			decoder_channels,
			block_depth=decoder_block_depth,
			block_kernel=decoder_block_kernel,
			block_expansion=decoder_block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
		)

		self.upsample_head = ProgressiveUpsampleHead(
			in_channels=decoder_channels[0],
			out_channels=out_channels,
			target_bins=out_bins,
			in_bands=in_bands,
			upsample_scales=upsample_scales,
			tfc_tdf_depth=upsample_tfc_tdf_depth,
			tfc_tdf_bn=upsample_tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)

	def forward(self, x: Tensor) -> Tensor:
		H, _W = x.shape[2:]

		enc_feats = self.backbone(x)

		h_ace_feats = self.hyperace(enc_feats)

		dec_feat = self.decoder(enc_feats, h_ace_feats)

		feat_time_restored: Tensor = F.interpolate(dec_feat, size=(H, dec_feat.shape[-1]), mode='bilinear', align_corners=False)

		return self.upsample_head(feat_time_restored)

