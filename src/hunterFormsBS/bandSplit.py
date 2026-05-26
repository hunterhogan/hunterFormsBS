"""Use band-splitting and training-loss utilities for source separation.

You can use this module to build the frequency-domain front end and the per-stem output heads shared
by BS-RoFormer [1] and Mel-Band RoFormer [2]. `BandSplit` projects grouped STFT frequency bins into a
common feature width. `MaskEstimator` converts band tokens back into complex subband masks. `MLP`
constructs the band-local affine blocks used inside `MaskEstimator`. `lossComputation` combines a
waveform-domain mean absolute error term with a multi-resolution complex-STFT term to produce the
training objective. `DEFAULT_FREQS_PER_BANDS` provides the standard non-overlapping frequency-bin
partition used by the BS-RoFormer front end.

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

References
----------
[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
	Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
[2] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
	Transcription https://arxiv.org/abs/2409.04702
"""
from __future__ import annotations

from einops import rearrange
from hunterFormsBS.hyperACE import SegmModel
from hunterFormsBS.theTypes import ParametersComputeLoss, ParametersSTFT
from torch import nn, Tensor, tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import default
from torch_einops_kit.scaleValues import RMSNorm
from typing import Literal, overload, TYPE_CHECKING
import torch
import torch.nn.functional as F

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

		outs: list[Tensor] = []
		for split_input, to_feature in zip(tuple_inputs, self.to_features, strict=True):
			split_output: Tensor = to_feature(split_input)
			outs.append(split_output)

		return torch.stack(outs, dim=-2)

@overload
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ParametersComputeLoss, *, return_loss_breakdown: Literal[True]) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
@overload
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ParametersComputeLoss, *, return_loss_breakdown: Literal[False] = False) -> Tensor: ...
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ParametersComputeLoss, *, return_loss_breakdown: bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
	"""Compute waveform and multi-resolution spectrogram loss for selected stems.

	You can use `lossComputation` to train one separator that reconstructs waveform audio and matches
	target spectrogram structure at multiple STFT window sizes. The loss family follows the
	BS-RoFormer objective [1] and the later Mel-RoFormer formulation family [2]. `lossComputation`
	selects the target stems indicated by `stem_ids`, trims `target` to the reconstructed sample
	count, computes one waveform-domain mean absolute error term, accumulates one complex-STFT mean
	absolute error term across the configured resolutions, and combines both terms into one total
	loss. `lossComputation` is the shared training-loss helper used by
	`hunterFormsBS.bandSplitRotator.BandSplitRotator` [3].

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
		`multi_stft_resolution_loss` is accumulated with `torch.nn.functional.l1_loss` [4] on the
		complex STFT tensors returned by `torch.stft` [5]. In PyTorch, `torch.nn.functional.l1_loss`
		is equivalent to `torch.abs(input - target).mean()` [4], so the implemented complex norm uses
		the complex modulus from `torch.abs` [6] rather than an explicit sum of separate real and
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
	[3] `hunterFormsBS.bandSplitRotator.BandSplitRotator`

	[4] torch.nn.functional.l1_loss - PyTorch
		https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.l1_loss.html
	[5] torch.stft - PyTorch
		https://docs.pytorch.org/docs/stable/generated/torch.stft.html
	[6] torch.abs - PyTorch
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

		multi_stft_resolution_loss += F.l1_loss(recon_Y, target_Y)

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
	embedding-projection family from Mel-RoFormer [3]. Each band head uses `MLP` [4] as the affine
	block before the final gate.

	Attributes
	----------
	dim_inputs : list[int]
		Width of each band-local output segment in the concatenated return value.
	to_freqs : ModuleList
		Per-band projection head collection. Each head maps one `dim`-wide band token to one
		band-local output segment whose width is given by `dim_inputs`.

	Implementation boundary
	-----------------------
	band-head operators : behavior
		`MaskEstimator` stores the band-local affine stack from `MLP` [4] and the final gated output
		stage [6]. Any upstream normalization such as `RMSNorm` [5] must happen before `MaskEstimator`
		receives band token `Tensor` `x`.
	package default head template : behavior
		When `depth = 1`, `mlp_expansion_factor = 4`, and `activation` is the hyperbolic tangent
		activation, `MaskEstimator` matches the default hidden-width and activation pattern used by the
		published model family [2][3][6].

	References
	----------
	[1] `hunterFormsBS.bandSplitRotator.BandSplitRotator`

	[2] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation
		with Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
	[3] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
		Transcription https://arxiv.org/abs/2409.04702

	[4] `MLP`

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
		[3] `hunterFormsBS.bandSplitRotator.BandSplitRotator`

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
			out += y  # pyright: ignore[reportPossiblyUnboundVariable]

		return out

def MLP(dim_in: int, dim_out: int, dim_hidden: int | None = None, depth: int = 1, activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
	"""Build one feedforward projection from `dim_in` to `dim_out`.

	You can use `MLP` to create one `nn.Sequential` made of linear layers with one intermediate
	activation after every non-final linear layer. `MLP` does not apply normalization or output gating
	by itself. `MaskEstimator` [1] uses `MLP` as the band-local affine block before the final gate in
	the BS-RoFormer [2] and Mel-RoFormer [3] mask heads.

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

		where  g ≜ affine output returned by `MLP` before the later gate in `MaskEstimator`
	```

	Layer Construction
	------------------
	activation instantiation : behavior
		`MLP` instantiates a fresh `activation()` module after each non-final linear layer. `MLP` does
		not append an activation after the last linear layer.

	References
	----------
	[1] `MaskEstimator`

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
