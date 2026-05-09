"""Use band-splitting, mask estimation, and training-loss utilities for source separation.

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
[2] Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation.
    https://arxiv.org/abs/2409.04702
"""
from __future__ import annotations

from collections.abc import Sequence
from einops import rearrange
from hunterFormsBS import ComputeLoss, KwargsSTFT
from torch import nn, Tensor, tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import default
from torch_einops_kit.scaleValues import RMSNorm
from typing import Literal, overload
import torch
import torch.nn.functional as F

DEFAULT_FREQS_PER_BANDS: tuple[int, ...] = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)

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
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ComputeLoss, *, return_loss_breakdown: Literal[True]) -> tuple[Tensor, tuple[Tensor, Tensor]]:...
@overload
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ComputeLoss, *, return_loss_breakdown: Literal[False] = False) -> Tensor:...
def lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ComputeLoss, *, return_loss_breakdown: bool = False) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
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
	multi_stft : ComputeLoss
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
		Let y ≜ `target_sel`, ŷ ≜ `recon_audio`, W ≜ `multi_stft['window_sizes']`,
			α ≜ `multi_stft['loss_weight']`

		L_time = ‖ŷ − y‖₁
		L_stft = ∑_{w ∈ W} ‖STFT_w(ŷ) − STFT_w(y)‖₁
		L_total = L_time + α · L_stft

		where  L_time ≜ `loss`,  L_stft ≜ `multi_stft_resolution_loss`,
			STFT_w(·) ≜ `torch.stft` for window size w,  L_total ≜ `total_loss`
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
	[2] Wang, J.-C., Lu, W.-T., Won, M., Choi, K., & Song, X. (2024). Mel-RoFormer for Vocal
		Separation and Vocal Melody Transcription. https://arxiv.org/abs/2409.04702
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
		res_stft_kwargs = KwargsSTFT(
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
	[3] usually reshapes the concatenated output into real and imaginary mask values and, when the
	band layout overlaps in frequency, averages repeated frequency-bin estimates outside this class.
	`MaskEstimator` corresponds to the multi-band mask-estimation family from BS-RoFormer [1] and the
	Embedding Projection family from Mel-RoFormer [2]. Each band head uses
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

	Architecture relation
	---------------------
	paper head boundary : behavior
		The papers describe each band head with `RMSNorm`, affine transforms, `tanh`, and `GLU`
		[1][2][5][6]. `MaskEstimator` implements the band-local affine and gating portion, while any
		upstream normalization happens before `MaskEstimator` is called. When `depth = 1`,
		`mlp_expansion_factor = 4`, and `activation` is the hyperbolic tangent activation,
		`MaskEstimator` matches the paper-default hidden-width pattern.
	overlap averaging outside this class : behavior
		`MaskEstimator` does not merge overlapping frequency bins by itself. In the Mel-RoFormer
		formulation [2], the later full-model step averages overlapping frequency-bin estimates after
		the band-local outputs are reshaped to the paper-level mask layout.

	References
	----------
	[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation
		with Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
	[2] Wang, J.-C., Lu, W.-T., Chen, J., Won, M., Choi, K., & Song, X. (2024).
		Mel-RoFormer for Vocal Separation and Vocal Melody Transcription.
		https://arxiv.org/abs/2409.04702
	[3] hunterFormsBS.bandSplitRotator.BandSplitRotator

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

		for dim_in in self.dim_inputs:
			mlp: nn.Sequential = nn.Sequential(MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth, activation=activation), nn.GLU(dim=-1))
			self.to_freqs.append(mlp)

	def forward(self, x: Tensor) -> Tensor:
		"""Estimate one concatenated mask from band features `x`.

		You can use `forward` when `x` contains one learned token per band on the penultimate axis.
		`forward` sends each band token to the matching projection head in `self.to_freqs`, then
		concatenates the band-local outputs along the last axis. The return value keeps band order but
		removes the explicit band axis, so callers can reinterpret contiguous output segments as one
		complex mask layout for later reshaping and scattering.

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

		Shape Transformation
		--------------------
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
		band-local projection family : equation
		```
			Let K ≜ len(`self.dim_inputs`),  zₖ ≜ `self.dim_inputs[k]`
				x = [x₀, …, xₖ]  for  k ∈ {0, …, K − 1}

			mₖ = Φₖ(xₖ) ∈ ℝ^{…, zₖ}
			m = [m₀ ‖ ⋯ ‖ mₖ]  for  k ∈ {0, …, K − 1}
		```
		paper-default band head : equation
		```
			Let  uₖ denote the band feature presented to the paper head

			uₖ = RMSNorm(xₖ)
			Φₖ(xₖ) = GLU(Wₖ,2 tanh(Wₖ,1 uₖ + bₖ,1) + bₖ,2)
			GLU([aₖ ‖ bₖ]) = aₖ ⊙ σ(bₖ)
		```

		Paper alignment
		---------------
		BS-RoFormer mask assembly : behavior
			In BS-RoFormer [1], the concatenated band outputs form the full cIRM by frequency-axis
			concatenation of the band-local masks.
		Mel-RoFormer mask assembly : behavior
			In Mel-RoFormer [2], the concatenated band outputs correspond to the list `[Φ̂⁰, Φ̂¹, …,
			Φ̂ᴷ⁻¹]` from equation (1), and the later full-model step computes `M̂[c, f, t] = (1 / S_f)
			∑_k Φ̂ᵏ[c, f, t]` from equation (2) for overlapping bins.
		normalization boundary : behavior
			`forward` itself does not apply per-band `RMSNorm`. Any upstream normalization must happen
			before `forward` is called [3][4].
		paper operators : behavior
			The paper-default head uses `RMSNorm` [4] before the affine block and `GLU` [5] as the
			final gate.

		References
		----------
		[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation
			with Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
		[2] Wang, J.-C., Lu, W.-T., Chen, J., Won, M., Choi, K., & Song, X. (2024).
			Mel-RoFormer for Vocal Separation and Vocal Melody Transcription.
			https://arxiv.org/abs/2409.04702
		[3] hunterFormsBS.bandSplitRotator.BandSplitRotator

		[4] Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
			https://papers.nips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html
		[5] Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language Modeling with
			Gated Convolutional Networks. https://proceedings.mlr.press/v70/dauphin17a.html
		"""
		tensor_unbound: tuple[Tensor, ...] = x.unbind(dim=-2)

		outs: list[Tensor] = []

		for band_features, mlp in zip(tensor_unbound, self.to_freqs, strict=True):
			freq_out: Tensor = mlp(band_features)
			outs.append(freq_out)

		return torch.cat(outs, dim=-1)

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
		Let d₀ ≜ `dim_in`,  d_out ≜ `dim_out`,  d₁ = ⋯ = d_L ≜ `dim_hidden`,
			L ≜ `depth`

		h₀ = x
		hᵢ₊₁ = φ(Wᵢ hᵢ + bᵢ)    for  i ∈ {0, …, L − 1}
		y = W_last h_L + b_last

		where  φ = `activation`
	```
	paper-default affine block : equation
	```
		For `hunterFormsBS.bandSplit.MaskEstimator` with `depth = 1`,
		`dim_hidden = 4 · dim_in`, and the hyperbolic tangent activation,

		h = tanh(W₁ x + b₁)
		g = W₂ h + b₂
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
	[3] Wang, J.-C., Lu, W.-T., Chen, J., Won, M., Choi, K., & Song, X. (2024).
		Mel-RoFormer for Vocal Separation and Vocal Melody Transcription.
		https://arxiv.org/abs/2409.04702
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
