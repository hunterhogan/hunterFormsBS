"""Estimate per-band complex subband masks for source separation.

You can use this module to build and apply band-local mask-estimation heads. `MaskEstimator`
converts one stack of band tokens into one concatenated band-local mask representation for a single
separator head. `MLP` constructs the affine blocks used inside `MaskEstimator`. The
mask-estimation family is shared by BS-RoFormer [1] and Mel-RoFormer [2] and is consumed by
`hunterFormsBS.bandSplitRotator.BandSplitRotator` [3].

Contents
--------
Functions
	MLP
		Build one feedforward projection sequence from an input width to an output width.

Classes
	MaskEstimator
		Estimate one concatenated subband mask from a stack of band tokens.

References
----------
[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
	Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
[2] Wang, J.-C., Lu, W.-T., and Chen, J. (2024) Mel-RoFormer for Vocal Separation and Vocal Melody
	Transcription https://arxiv.org/abs/2409.04702
[3] `hunterFormsBS.bandSplitRotator.BandSplitRotator`
"""
from __future__ import annotations

from einops import rearrange
from hunterFormsBS.hyperACE import SegmModel
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import default
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
	from collections.abc import Sequence

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
