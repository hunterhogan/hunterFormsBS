# ruff: noqa: E741
"""Assemble a HyperACE-guided spectrogram branch for band-wise mask estimation.

You can use this module to build the optional spectrogram branch that augments
`hunterFormsBS.bandSplit.MaskEstimator` [1] with a segmentation-style network. The module adapts the
Hypergraph-based Adaptive Correlation Enhancement (HyperACE) and FullPAD ideas from YOLOv13 [2] to
spectrogram features produced in the BS-RoFormer and Mel-RoFormer model family [3][4]. It also uses
TFC-TDF residual refinement blocks from the music-source-separation literature [5][6].

Contents
--------
Functions
	autopad
		Compute same-style padding for scalar or per-axis kernel sizes.

Classes
	AdaptiveHyperedgeGeneration
		Generate a continuous vertex-to-hyperedge participation matrix from feature tokens.
	AdaptiveHypergraphComputation
		Apply adaptive hypergraph computation to one spatial feature map.
	Backbone
		Encode one spectrogram-like feature map into four progressively coarser stages.
	C3AH
		Combine a CSP-style split with adaptive hypergraph computation.
	Conv
		Apply one convolution-normalization-activation block to 2D features.
	Decoder
		Fuse encoder stages with HyperACE features and decode them toward the input resolution.
	DS_Bottleneck
		Apply two depthwise-separable convolution blocks with an optional residual shortcut.
	DS_C3k
		Apply a CSP-style depthwise-separable feature block.
	DS_C3k2
		Apply a lightweight C3k-style refinement block with one inner DS_C3k stage.
	DSConv
		Apply one depthwise-separable convolution-normalization-activation block to 2D features.
	FreqPixelShuffle
		Expand frequency resolution with a channel-to-frequency rearrangement.
	GatedFusion
		Fuse one decoder feature map with one resized HyperACE feature map.
	HyperACE
		Aggregate multi-scale encoder features with high-order and low-order branches.
	HypergraphConvolution
		Propagate features through hyperedges and back to vertices.
	ProgressiveUpsampleHead
		Recover target frequency resolution through repeated frequency-axis upsampling.
	SegmModel
		Assemble the full segmentation-style HyperACE branch used by the mask estimator.
	TFC_TDF
		Refine one feature map with residual TFC-TDF blocks.

References
----------
[1] hunterFormsBS.bandSplit.MaskEstimator

[2] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
	and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual
	Perception. https://arxiv.org/abs/2506.17733
[3] Lu, W.-T., Wang, J.-C., Kong, Q., and Hung, Y.-N. (2023). Music Source Separation
	with Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
[4] Wang, J.-C., Lu, W.-T., and Chen, J. (2024). Mel-RoFormer for Vocal Separation and
	Vocal Melody Transcription. https://arxiv.org/abs/2409.04702
[5] Chen, J., Wu, L., Suryotrisongko, H., and Kim, M. (2024). Music source separation
	based on a lightweight deep learning framework (DTTNET: DUAL-PATH TFC-TDF UNET).
	https://arxiv.org/abs/2309.08684
[6] Kim, M., Lee, J. H., and Jung, S. (2023). Sound Demixing Challenge 2023 Music
	Demixing Track Technical Report: TFC-TDF-UNet v3. https://arxiv.org/abs/2306.09382
"""
from __future__ import annotations

from more_itertools import loops
from torch import nn, Tensor
from typing import cast, overload, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
	from collections.abc import Callable

class Conv(nn.Module):
	"""Apply one convolution-normalization-activation block to 2D features.

	(AI generated docstring)

	You can use `Conv` when one feature map needs one 2D convolution, one instance normalization
	layer, and one activation layer in sequence [1]. `Conv` computes same-style padding with `autopad`
	[2], applies the convolution, normalizes the projected features, and then applies the chosen
	activation unless `act` disables it.

	Mathematics
	-----------
	convolution block : equation
	```
		Let X ≜ `x`,  C ≜ `self.conv`,  N ≜ `self.bn`,  σ ≜ `self.act`

		Y = σ(N(C(X)))

		where Y ≜ `transformedTensor`
	```

	References
	----------
	[1] PyTorch.
		https://context7.com/pytorch/pytorch
	[2] `autopad`
	"""

	def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, act: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		"""Configure one convolution-normalization-activation block.

		(AI generated docstring)

		You can use `__init__` to choose the input width, output width, kernel size, stride, group
		count, and activation family for one `Conv` block. `__init__` stores one `nn.Conv2d`, one
		`nn.InstanceNorm2d`, and one activation module.

		Parameters
		----------
		c1 : int
			Input channel count.
		c2 : int
			Output channel count.
		k : int = 1
			Convolution kernel size.
		s : int = 1
			Convolution stride.
		p : int | None = None
			Explicit padding. When `p is None`, `autopad` computes same-style padding [1].
		g : int = 1
			Convolution group count.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated after normalization when `act` is `True`.
		norm_eps : float = 1e-8
			Epsilon passed to `nn.InstanceNorm2d` [2].
		act : bool = True
			Whether to instantiate `activation` instead of `nn.Identity` [2].
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		bias : bool = False
			Whether `nn.Conv2d` learns a bias term.

		References
		----------
		[1] `autopad`

		[2] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		super().__init__()
		self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bias)
		self.bn = nn.InstanceNorm2d(c2, affine=norm_affine, eps=norm_eps)
		self.act: nn.Module = activation() if act else nn.Identity()

	def forward(self, x: Tensor) -> Tensor:
		"""Transform one 2D feature tensor with the stored convolution block.

		(AI generated docstring)

		You can use `forward` to map one feature tensor `x` through the stored convolution,
		instance-normalization, and activation layers in `Conv`.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, c1, height, width)`.

		Returns
		-------
		transformedTensor : Tensor
			Output tensor of shape `(batch, c2, height_out, width_out)`.

		PyTorch
		-------
		tensor layout : `nn.Conv2d` and `nn.InstanceNorm2d`
			`forward` expects `x` in `NCHW` layout because `Conv` stores one `nn.Conv2d` and one
			`nn.InstanceNorm2d` [1]. The spatial size of `transformedTensor` follows the stride and
			padding stored in `self.conv`.

		See Also
		--------
		Conv
			Store the layers and parameter choices used by `forward`.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		return self.act(self.bn(self.conv(x)))

@overload
def autopad(k: int, p: int | None = None) -> int: ...
@overload
def autopad(k: list[int], p: list[int] | None = None) -> list[int]: ...
def autopad(k: int | list[int], p: int | list[int] | None = None) -> int | list[int]:
	"""Compute same-style padding for one kernel specification.

	(AI generated docstring)

	You can use `autopad` to derive one padding specification from one kernel specification when a
	convolution block should preserve spatial size under unit stride. If `p` is already provided,
	`autopad` returns it unchanged. Otherwise `autopad` returns `k // 2` for a scalar kernel size or
	the per-axis half-width list for a list-valued kernel size.

	Parameters
	----------
	k : int | list[int]
		Kernel size specification.
	p : int | list[int] | None = None
		Explicit padding specification. When provided, `autopad` returns `p` unchanged.

	Returns
	-------
	padding : int | list[int]
		Padding specification compatible with `k`.
	"""
	if p is None:
		if isinstance(k, int):
			p = k // 2
		else:
			p = [x // 2 for x in k]
	return p

class DSConv(nn.Module):
	"""Apply one depthwise-separable convolution block to 2D features.

	(AI generated docstring)

	You can use `DSConv` when one feature map should be processed by one depthwise convolution, one
	pointwise convolution, one instance normalization layer, and one activation layer [1]. The block
	matches the depthwise-separable building unit used in the YOLOv13 DS-series design [2].

	Mathematics
	-----------
	depthwise-separable block [2 at Equation (14)] : equation
	```
		Let X ≜ `x`,  D ≜ `self.dwconv`,  P ≜ `self.pwconv`,
			N ≜ `self.bn`,  σ ≜ `self.act`

		Y = σ(N(P(D(X))))

		where Y ≜ `transformedTensor`
	```

	References
	----------
	[1] PyTorch.
		https://context7.com/pytorch/pytorch
	[2] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(self, c1: int, c2: int, k: int = 3, s: int | tuple[int, int] = 1, p: int | None = None
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			# TODO Why not `activation: type[nn.Module] | None`? or `activation: type[nn.Module] = nn.Identity`?
			, act: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		"""Configure one depthwise-separable convolution block.

		(AI generated docstring)

		You can use `__init__` to choose the channel counts, kernel size, stride, explicit padding,
		and activation family for one `DSConv` block.

		Parameters
		----------
		c1 : int
			Input channel count.
		c2 : int
			Output channel count.
		k : int = 3
			Depthwise kernel size.
		s : int | tuple[int, int] = 1
			Depthwise stride.
		p : int | None = None
			Explicit depthwise padding. When `p is None`, `autopad` computes same-style padding [1].
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated after normalization when `act` is `True`.
		norm_eps : float = 1e-8
			Epsilon passed to `nn.InstanceNorm2d` [2].
		act : bool = True
			Whether to instantiate `activation` instead of `nn.Identity` [2].
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		bias : bool = False
			Whether the depthwise and pointwise convolutions learn bias terms.

		References
		----------
		[1] `autopad`

		[2] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		super().__init__()
		self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=bias)
		self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=bias)
		self.bn = nn.InstanceNorm2d(c2, affine=norm_affine, eps=norm_eps)
		self.act: nn.Module = activation() if act else nn.Identity()

	def forward(self, x: Tensor) -> Tensor:
		"""Transform one 2D feature tensor with depthwise-separable convolution.

		(AI generated docstring)

		You can use `forward` to apply the depthwise path, pointwise projection, instance
		normalization, and activation stored in `DSConv`.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, c1, height, width)`.

		Returns
		-------
		transformedTensor : Tensor
			Output tensor of shape `(batch, c2, height_out, width_out)`.

		PyTorch
		-------
		depthwise-separable layout : grouped convolution followed by `1 × 1` projection
			`forward` applies one grouped `nn.Conv2d` with `groups == c1`, one pointwise `nn.Conv2d`,
			one `nn.InstanceNorm2d`, and one activation module in `NCHW` layout [1].

		See Also
		--------
		DSConv
			Store the layers and parameter choices used by `forward`.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		return self.act(self.bn(self.pwconv(self.dwconv(x))))

class DS_Bottleneck(nn.Module):
	"""Apply two depthwise-separable convolution blocks with an optional residual shortcut.

	(AI generated docstring)

	You can use `DS_Bottleneck` to stack one fixed `3 × 3` depthwise-separable block with one
	configurable `k × k` depthwise-separable block [1]. When the input and output channel counts match
	and `shortcut` is `True`, `DS_Bottleneck` adds the input feature map back to the output.

	Mathematics
	-----------
	depthwise bottleneck [1 at Equation (15)] : equation
	```
		Let X ≜ `x`,  D₁ ≜ `self.dsconv1`,  D₂ ≜ `self.dsconv2`,
			𝟙ₛ ≜ 1_{`self.shortcut`}

		Z = D₂(D₁(X))
		Y = Z + 𝟙ₛ X

		where Y ≜ `refinedTensor`
	```

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(self, c1: int, c2: int, k: int = 3
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, shortcut: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		"""Configure one depthwise-separable bottleneck block.

		(AI generated docstring)

		You can use `__init__` to choose the channel counts, large-kernel size, residual-shortcut
		behavior, and activation family for one `DS_Bottleneck` block.

		Parameters
		----------
		c1 : int
			Input channel count.
		c2 : int
			Output channel count.
		k : int = 3
			Kernel size used by the second depthwise-separable block.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in both inner `DSConv` blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in both inner `DSConv` blocks.
		shortcut : bool = True
			Whether to add the input feature map back when `c1 == c2`.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters in both inner blocks.
		bias : bool = False
			Whether the inner convolutions learn bias terms.
		"""
		super().__init__()
		c_: int = c1
		self.dsconv1: DSConv = DSConv(c1, c_, k=3, s=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.dsconv2: DSConv = DSConv(c_, c2, k=k, s=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.shortcut: bool = shortcut and c1 == c2

	def forward(self, x: Tensor) -> Tensor:
		"""Refine one 2D feature tensor with a two-step DS bottleneck.

		(AI generated docstring)

		You can use `forward` to pass one feature tensor `x` through the two stored `DSConv` blocks
		and, when `self.shortcut` is enabled, add `x` back to the refined tensor.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, c1, height, width)`.

		Returns
		-------
		refinedTensor : Tensor
			Output tensor of shape `(batch, c2, height, width)`.

		PyTorch
		-------
		residual condition : channel-match shortcut
			`forward` applies the residual addition only when `self.shortcut` is `True`, which
			requires the channel counts chosen in `DS_Bottleneck.__init__` to make `x` and the refined
			tensor shape-compatible.

		See Also
		--------
		DS_Bottleneck
			Store the two inner `DSConv` blocks and the shortcut rule.

		References
		----------
		[1] `DSConv`
		"""
		return x + self.dsconv2(self.dsconv1(x)) if self.shortcut else self.dsconv2(self.dsconv1(x))

class DS_C3k(nn.Module):
	"""Apply a CSP-style depthwise-separable feature block.

	(AI generated docstring)

	You can use `DS_C3k` to split one feature map into one transformed branch and one lateral branch,
	process the transformed branch with repeated `DS_Bottleneck` blocks, concatenate both branches,
	and fuse them back to the requested output width [1].

	Mathematics
	-----------
	CSP depthwise branch [1 at Figure 4] : equation
	```
		Let X ≜ `x`,  C₁ ≜ `self.cv1`,  C₂ ≜ `self.cv2`,
			C₃ ≜ `self.cv3`,  B ≜ `self.m`

		Y = C₃([B(C₁(X)) ‖ C₂(X)]_channel)

		where Y ≜ `fusedTensor`
	```

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, norm_affine: bool = True
			, bias: bool = False) -> None:
		"""Configure one `DS_C3k` block.

		(AI generated docstring)

		You can use `__init__` to choose the input width, output width, bottleneck repeat count,
		large-kernel size, and expansion ratio for one `DS_C3k` block.

		Parameters
		----------
		c1 : int
			Input channel count.
		c2 : int
			Output channel count.
		n : int = 1
			Number of inner `DS_Bottleneck` blocks.
		k : int = 3
			Kernel size used by each inner `DS_Bottleneck` block.
		e : float = 0.5
			Expansion ratio used to compute the hidden width `int(c2 * e)`.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in all inner blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in all inner blocks.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		bias : bool = False
			Whether the inner convolutions learn bias terms.
		"""
		super().__init__()
		c_ = int(c2 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv2: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv3: Conv = Conv(2 * c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.m = nn.Sequential(
			*[
				DS_Bottleneck(c_, c_, k=k, shortcut=True, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
				for _index in loops(n)
			]
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Fuse one lateral branch with one repeated DS bottleneck branch.

		(AI generated docstring)

		You can use `forward` to split one feature tensor `x` into a transformed branch and a lateral
		branch, concatenate both branches, and fuse the result back to the requested output width.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, c1, height, width)`.

		Returns
		-------
		fusedTensor : Tensor
			Output tensor of shape `(batch, c2, height, width)`.

		PyTorch
		-------
		channel fusion : `torch.cat(..., dim=1)`
			`forward` keeps the spatial axes unchanged and concatenates the transformed branch with
			the lateral branch along channel axis `1` before the final `1 × 1` fusion convolution [1].

		See Also
		--------
		DS_C3k
			Store the branch projections and repeated `DS_Bottleneck` stack.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class DS_C3k2(nn.Module):
	"""Apply a lightweight C3k-style refinement block with one inner `DS_C3k` stage.

	(AI generated docstring)

	You can use `DS_C3k2` when one feature map should first be projected to one hidden width, then
	refined by one `DS_C3k` block, and finally fused back to the requested output width [1].

	Mathematics
	-----------
	C3k2 refinement [1 at Figure 4] : equation
	```
		Let X ≜ `x`,  C₁ ≜ `self.cv1`,  B ≜ `self.m`,  C₂ ≜ `self.cv2`

		Z = C₁(X)
		Y = C₂(B(Z))

		where Y ≜ `refinedTensor`
	```

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, norm_affine: bool = True
			, bias: bool = False) -> None:
		"""Configure one `DS_C3k2` block.

		(AI generated docstring)

		You can use `__init__` to choose the input width, output width, inner repeat count,
		large-kernel size, and expansion ratio for one `DS_C3k2` block.

		Parameters
		----------
		c1 : int
			Input channel count.
		c2 : int
			Output channel count.
		n : int = 1
			Repeat count used inside the inner `DS_C3k` block.
		k : int = 3
			Kernel size used inside the inner `DS_C3k` block.
		e : float = 0.5
			Expansion ratio used to compute the hidden width `int(c2 * e)`.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in all inner blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in all inner blocks.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		bias : bool = False
			Whether the inner convolutions learn bias terms.
		"""
		super().__init__()
		c_ = int(c2 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.m: DS_C3k = DS_C3k(c_, c_, n=n, k=k, e=1.0, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv2: Conv = Conv(c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)

	def forward(self, x: Tensor) -> Tensor:
		"""Project, refine, and reproject one 2D feature tensor.

		(AI generated docstring)

		You can use `forward` to reduce the channel width of `x`, refine the reduced feature tensor
		with the inner `DS_C3k` block, and project the refined tensor to the requested output width.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, c1, height, width)`.

		Returns
		-------
		refinedTensor : Tensor
			Output tensor of shape `(batch, c2, height, width)`.

		See Also
		--------
		DS_C3k2
			Store the projection, refinement, and reprojection submodules.
		DS_C3k
			Implement the inner refinement stage.
		"""
		x_ = self.cv1(x)
		x_ = self.m(x_)
		return self.cv2(x_)

class AdaptiveHyperedgeGeneration(nn.Module):
	"""Generate a continuous vertex-to-hyperedge participation matrix from feature tokens.

	(AI generated docstring)

	You can use `AdaptiveHyperedgeGeneration` to summarize one spatial feature map into a set of
	context-conditioned hyperedge prototypes and then compute soft vertex-to-hyperedge assignments for
	every token [1]. The returned participation matrix is the adaptive incidence-like matrix used by
	HyperACE before hypergraph message passing.

	Mathematics
	-----------
	adaptive participation matrix [1 at Equations (1)-(4)] : equation
	```
		Let X ≜ `x`,  B ≜ |X|₀,  N ≜ |X|₁,  C ≜ |X|₂,
			M ≜ `self.num_hyperedges`,  H ≜ `self.num_heads`,  d ≜ C / H,
			P⁰ ≜ `self.global_proto`,  Φ ≜ `self.context_mapper`,
			Wᵠ ≜ `self.query_proj`

		fᵃᵛᵍ[b, c] = (1 / N) ∑ᵢ X[b, i, c]
		fᵐᵃˣ[b, c] = maxᵢ X[b, i, c]
		fᶜᵗˣ[b] = [fᵃᵛᵍ[b] ‖ fᵐᵃˣ[b]]
		P[b, m, :] = P⁰[m, :] + reshape_M×C(Φ(fᶜᵗˣ[b]))[m, :]
		Z[b, i, :] = Wᵠ X[b, i, :]
		Sʰ[b, i, m] = ⟨Zʰ[b, i, :], Pʰ[b, m, :]⟩ / √d
		S̄[b, i, m] = (1 / H) ∑ₕ Sʰ[b, i, m]
		A[b, i, m] = exp(S̄[b, i, m]) / ∑ⱼ exp(S̄[b, j, m])
		Aᵗ[b, m, i] = A[b, i, m]

		where Aᵗ ≜ `participationMatrix`
	```

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(self, in_channels: int, num_hyperedges: int, num_heads: int = 8, *, linear_bias: bool = False) -> None:
		"""Configure one adaptive hyperedge generator.

		(AI generated docstring)

		You can use `__init__` to choose the token width, the number of learned hyperedges, the number
		of attention-style heads, and whether the internal linear layers learn bias terms.

		Parameters
		----------
		in_channels : int
			Token width `C`. `in_channels` should be divisible by `num_heads` because the projection
			is reshaped into `num_heads` heads of width `C / num_heads`.
		num_hyperedges : int
			Number of adaptive hyperedges `M` used to summarize the token set.
		num_heads : int = 8
			Number of heads used to compute multi-head token-to-prototype similarity.
		linear_bias : bool = False
			Whether `context_mapper` and `query_proj` learn bias terms.
		"""
		super().__init__()
		self.num_hyperedges: int = num_hyperedges
		self.num_heads: int = num_heads
		self.head_dim: int = in_channels // num_heads

		self.global_proto: nn.Parameter = nn.Parameter(torch.randn(num_hyperedges, in_channels))

		self.context_mapper: nn.Linear = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=linear_bias)

		self.query_proj: nn.Linear = nn.Linear(in_channels, in_channels, bias=linear_bias)

		self.scale: float = self.head_dim**-0.5

	def forward(self, x: Tensor) -> Tensor:
		"""Return adaptive hyperedge participation weights for one token set.

		(AI generated docstring)

		You can use `forward` to summarize one token tensor `x` into context-dependent hyperedge
		prototypes and then compute one soft participation tensor that tells how strongly each token
		participates in each hyperedge.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, token_count, in_channels)`.

		Returns
		-------
		participationMatrix : Tensor
			Tensor of shape `(batch, num_hyperedges, token_count)`.

		PyTorch
		-------
		returned layout : `torch.bmm`-ready storage
			The HyperACE paper writes participation matrix `A` with vertex index first [1]. This
			implementation permutes the result to `(batch, num_hyperedges, token_count)` so that
			`torch.bmm(participationMatrix, x)` can aggregate one hyperedge feature tensor directly in
			`HypergraphConvolution` [2].

		See Also
		--------
		AdaptiveHyperedgeGeneration
			Store the learnable prototypes and projection layers used by `forward`.
		HypergraphConvolution
			Consume `participationMatrix` for hypergraph message passing.

		References
		----------
		[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
			and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
			Visual Perception. https://arxiv.org/abs/2506.17733
		[2] `HypergraphConvolution`
		"""
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
	"""Propagate features through hyperedges and back to vertices.

	(AI generated docstring)

	You can use `HypergraphConvolution` after `AdaptiveHyperedgeGeneration` has produced a soft
	vertex-to-hyperedge participation matrix. The block first aggregates vertex features into
	hyperedge features and then projects those hyperedge features back to vertex space [1].

	Mathematics
	-----------
	hypergraph message passing [1 at Equation (5)] : equation
	```
		Let X ≜ `x`,  Aᵗ ≜ `A`,  Wᵉ ≜ `self.W_e`,
			Wᵛ ≜ `self.W_v`,  σ ≜ `self.act`,
			A[b, i, m] ≜ Aᵗ[b, m, i]

		F[b, m, :] = σ(Wᵉ(∑ᵢ A[b, i, m] X[b, i, :]))
		X̂[b, i, :] = σ(Wᵛ(∑ₘ A[b, i, m] F[b, m, :]))
		Y = X + X̂

		where Y ≜ `updatedTensor`
	```

	See Also
	--------
	AdaptiveHyperedgeGeneration
		Produce the participation matrix consumed by `HypergraphConvolution`.

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(self, in_channels: int, out_channels: int, activation: type[nn.Module] = nn.SiLU, *, linear_bias: bool = False) -> None:
		"""Configure one hypergraph convolution block.

		(AI generated docstring)

		You can use `__init__` to choose the input width, output width, activation family, and bias
		behavior for one `HypergraphConvolution` block.

		Parameters
		----------
		in_channels : int
			Input vertex-feature width.
		out_channels : int
			Output vertex-feature width produced by `W_v`. In the current residual implementation,
			`out_channels` should match `in_channels` so that `x + x̂` is well-defined.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated after both linear projections.
		linear_bias : bool = False
			Whether `W_e` and `W_v` learn bias terms.
		"""
		super().__init__()
		self.W_e = nn.Linear(in_channels, in_channels, bias=linear_bias)
		self.W_v = nn.Linear(in_channels, out_channels, bias=linear_bias)
		self.act: nn.Module = activation()

	def forward(self, x: Tensor, A: Tensor) -> Tensor:
		"""Propagate one token set through hyperedges and back to vertices.

		(AI generated docstring)

		You can use `forward` to aggregate token features from `x` into hyperedge features using `A`,
		project those hyperedge features back to token space, and add the projected update to `x`
		residually.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, token_count, in_channels)`.
		A : Tensor
			Participation tensor of shape `(batch, num_hyperedges, token_count)`.

		Returns
		-------
		updatedTensor : Tensor
			Tensor of shape `(batch, token_count, out_channels)`.

		PyTorch
		-------
		batched matrix products : `torch.bmm`
			`forward` uses `torch.bmm(A, x)` to gather one hyperedge feature tensor and uses
			`torch.bmm(A.transpose(1, 2), f_m)` to distribute the hyperedge features back to token
			space [1]. When `out_channels == in_channels`, the residual addition `x + x_out` is
			shape-compatible.

		See Also
		--------
		AdaptiveHyperedgeGeneration
			Produce the participation tensor consumed by `forward`.
		HypergraphConvolution
			Store the projection layers and activation used by `forward`.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		f_m: Tensor = torch.bmm(A, x)
		f_m = self.act(self.W_e(f_m))

		x_out: Tensor = torch.bmm(A.transpose(1, 2), f_m)
		x_out = self.act(self.W_v(x_out))

		return x + x_out

class AdaptiveHypergraphComputation(nn.Module):
	"""Apply adaptive hypergraph computation to one spatial feature map.

	(AI generated docstring)

	You can use `AdaptiveHypergraphComputation` to flatten one spatial feature map into tokens,
	compute adaptive hyperedge assignments with `AdaptiveHyperedgeGeneration`, apply
	`HypergraphConvolution`, and then restore the original spatial layout.

	Mathematics
	-----------
	spatial adaptive hypergraph [1 at Equations (1)-(5)] : transformation
	```
		Let X ≜ `x`,  B ≜ |X|₀,  C ≜ |X|₁,  Hₛ ≜ |X|₂,  Wₛ ≜ |X|₃,
			G ≜ `self.adaptive_hyperedge_gen`,  H ≜ `self.hypergraph_conv`

		Xᵗ[b, i, c] = X[b, c, r, q],  i = rWₛ + q
		Aᵗ = G(Xᵗ)
		Z = H(Xᵗ, Aᵗ)
		Y[b, c, r, q] = Z[b, rWₛ + q, c]

		where Y ≜ `refinedTensor`
	```

	See Also
	--------
	AdaptiveHyperedgeGeneration
		Generate the adaptive participation matrix.
	HypergraphConvolution
		Apply hypergraph message passing with that participation matrix.

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(self, in_channels: int, out_channels: int, num_hyperedges: int = 8, num_heads: int = 8
			, activation: type[nn.Module] = nn.SiLU
			, *
			, linear_bias: bool = False) -> None:
		"""Configure one spatial adaptive-hypergraph block.

		(AI generated docstring)

		You can use `__init__` to choose the token widths, the number of hyperedges, the number of
		heads, and the activation family for one `AdaptiveHypergraphComputation` block.

		Parameters
		----------
		in_channels : int
			Input channel count before flattening.
		out_channels : int
			Output channel count after hypergraph propagation.
		num_hyperedges : int = 8
			Number of adaptive hyperedges used in `AdaptiveHyperedgeGeneration`.
		num_heads : int = 8
			Number of heads used in `AdaptiveHyperedgeGeneration`.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated inside `HypergraphConvolution`.
		linear_bias : bool = False
			Whether the internal linear layers learn bias terms.
		"""
		super().__init__()
		self.adaptive_hyperedge_gen = AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads, linear_bias=linear_bias)
		self.hypergraph_conv = HypergraphConvolution(in_channels, out_channels, activation=activation, linear_bias=linear_bias)

	def forward(self, x: Tensor) -> Tensor:
		"""Apply adaptive hypergraph computation to one 2D feature tensor.

		(AI generated docstring)

		You can use `forward` to flatten one spatial feature tensor `x` into tokens, compute adaptive
		hyperedge participation, propagate information through the hypergraph block, and restore the
		original spatial layout.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, in_channels, height, width)`.

		Returns
		-------
		refinedTensor : Tensor
			Output tensor of shape `(batch, out_channels, height, width)`.

		PyTorch
		-------
		flatten-and-restore path : token view of one feature map
			`forward` reshapes `x` from `NCHW` layout to `(batch, token_count, in_channels)` with
			`token_count == height × width`, applies the adaptive hypergraph block, and then restores
			`NCHW` layout by `permute` and `view` [1].

		See Also
		--------
		AdaptiveHyperedgeGeneration
			Compute the adaptive participation tensor.
		HypergraphConvolution
			Apply message passing in token space.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		B, _C, H, W = x.shape
		x_flat: Tensor = x.flatten(2).permute(0, 2, 1)

		A = self.adaptive_hyperedge_gen(x_flat)

		x_out_flat = self.hypergraph_conv(x_flat, A)

		return x_out_flat.permute(0, 2, 1).view(B, -1, H, W)

class C3AH(nn.Module):
	"""Combine a CSP-style split with adaptive hypergraph computation.

	(AI generated docstring)

	You can use `C3AH` as the high-order correlation block inside `HyperACE`. `C3AH` splits one
	feature map into one lateral branch and one adaptive-hypergraph branch, processes the second
	branch with `AdaptiveHypergraphComputation`, concatenates both branches, and fuses them with a
	final `1 × 1` projection [1].

	Mathematics
	-----------
	C3AH split projection [1 at Equation (6)] : equation
	```
		Let X₀ ≜ `x`,  C₁ ≜ `self.cv1`,  C₂ ≜ `self.cv2`

		Xˡ = C₁(X₀)
		Xᵃ = C₂(X₀)
	```

	C3AH adaptive fusion [1 at Equation (7)] : equation
	```
		Let X₀ ≜ `x`,  C₁ ≜ `self.cv1`,  C₂ ≜ `self.cv2`,
			H ≜ `self.ahc`,  C₃ ≜ `self.cv3`

		Xˡ = C₁(X₀)
		Xᵃ = C₂(X₀)
		Xʰ = H(Xᵃ)
		Y = C₃([Xʰ ‖ Xˡ]_channel)

		where Y ≜ `fusedTensor`
	```

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(
		self
		, c1: int
		, c2: int
		, num_hyperedges: int = 8
		, num_heads: int = 8
		, e: float = 0.5
		, activation: type[nn.Module] = nn.SiLU
		, norm_eps: float = 1e-8
		, *
		, conv_bias: bool = False
		, linear_bias: bool = False
		, norm_affine: bool = True
	) -> None:
		"""Configure one `C3AH` block.

		(AI generated docstring)

		You can use `__init__` to choose the input width, output width, hypergraph size, expansion
		ratio, and normalization or bias settings for one `C3AH` block.

		Parameters
		----------
		c1 : int
			Input channel count.
		c2 : int
			Output channel count after the final fusion projection.
		num_hyperedges : int = 8
			Number of adaptive hyperedges used inside `AdaptiveHypergraphComputation`.
		num_heads : int = 8
			Number of heads used inside `AdaptiveHyperedgeGeneration`.
		e : float = 0.5
			Expansion ratio used to compute the hidden branch width `int(c1 * e)`.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in all inner blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in the inner `Conv` blocks.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		conv_bias : bool = False
			Whether the inner convolution layers learn bias terms.
		linear_bias : bool = False
			Whether the linear layers in `AdaptiveHypergraphComputation` learn bias terms.
		"""
		super().__init__()
		c_ = int(c1 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.cv2: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.ahc = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads, activation=activation, linear_bias=linear_bias)
		self.cv3: Conv = Conv(2 * c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)

	def forward(self, x: Tensor) -> Tensor:
		"""Fuse one lateral branch with one adaptive-hypergraph branch.

		(AI generated docstring)

		You can use `forward` to split one feature tensor `x` into a lateral branch and an
		adaptive-hypergraph branch, refine the second branch with `AdaptiveHypergraphComputation`, and
		fuse both branches by channel concatenation and one final `1 × 1` projection.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, c1, height, width)`.

		Returns
		-------
		fusedTensor : Tensor
			Output tensor of shape `(batch, c2, height, width)`.

		See Also
		--------
		AdaptiveHypergraphComputation
			Implement the adaptive-hypergraph branch.
		C3AH
			Store the branch projections and final fusion convolution.
		"""
		x_lateral = self.cv1(x)
		x_ahc = self.ahc(self.cv2(x))
		return self.cv3(torch.cat((x_ahc, x_lateral), dim=1))

class HyperACE(nn.Module):
	"""Aggregate multi-scale encoder features with high-order and low-order branches.

	(AI generated docstring)

	You can use `HyperACE` to fuse multiple encoder stages, split the fused tensor into high-order,
	low-order, and shortcut channels, refine those branches independently, and fuse them back into one
	decoder-conditioning feature map [1]. In this audio adaptation, the block consumes four encoder
	stages rather than the three backbone stages used in the vision paper.

	Mathematics
	-----------
	audio stage fusion : transformation
	```
		Let [B₂, B₃, B₄, B₅] ≜ `x`,  Rᵢ ≜ Resize(· → size(B₄)),
			Cᶠ ≜ `self.fuse_conv`

		Xᵇ = Cᶠ([R₂(B₂) ‖ R₃(B₃) ‖ B₄ ‖ R₅(B₅)]_channel)
		[Xᵇʰ, Xᵇˡ, Xˢ] = split_channel(Xᵇ; `self.c_h`, `self.c_l`, `self.c_s`)
	```

	high-order branches [1 at Equation (8)] : equation
	```
		Let K ≜ |`self.high_order_branch`|,
			Hₖ ≜ `self.high_order_branch[k]`,  k ∈ {0, …, K−1}

		Xʰₖ = Hₖ(Xᵇʰ)
	```

	high-order fusion [1 at Equation (9)] : equation
	```
		Let Cʰ ≜ `self.high_order_fuse`

		Xʰ = Cʰ([Xʰ₀ ‖ ⋯ ‖ Xʰ_{K−1}]_channel)
	```

	low-order branch [1 at Equation (10)] : equation
	```
		Let L ≜ `self.low_order_branch`

		Xˡ = L(Xᵇˡ)
	```

	final fusion [1 at Equation (11)] : equation
	```
		Let Cʸ ≜ `self.final_fuse`

		Y = Cʸ([Xʰ ‖ Xˡ ‖ Xˢ]_channel)

		where Y ≜ `conditioningTensor`
	```

	Implementation boundary
	-----------------------
	audio-stage adaptation : one fused spectrogram stage
		YOLOv13 applies HyperACE to three image-backbone stages [1]. This implementation first resizes
		four encoder stages `B2`, `B3`, `B4`, and `B5` to the spatial resolution of `B4`, fuses them
		with one `1 × 1` convolution, and then applies the same branch decomposition.

	See Also
	--------
	C3AH
		Implement the high-order branch block.
	DS_C3k
		Implement the low-order branch block.

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(
		self
		, in_channels: list[int]
		, out_channels: int
		, num_hyperedges: int = 8
		, num_heads: int = 8
		, k: int = 2
		, l: int = 1
		, c_h: float = 0.5
		, c_l: float = 0.25
		, c3ah_expansion: float = 1.0
		, low_order_depth: int = 1
		, low_order_kernel: int = 3
		, low_order_expansion: float = 1.0
		, activation: type[nn.Module] = nn.SiLU
		, norm_eps: float = 1e-8
		, *
		, conv_bias: bool = False
		, linear_bias: bool = False
		, norm_affine: bool = True
	) -> None:
		"""Configure one HyperACE aggregation block.

		(AI generated docstring)

		You can use `__init__` to choose the encoder-stage widths, output width, adaptive-hypergraph
		size, branch depths, branch width ratios, and normalization or bias settings for one
		`HyperACE` block.

		Parameters
		----------
		in_channels : list[int]
			Channel counts for the four encoder stages `[B2, B3, B4, B5]`.
		out_channels : int
			Output channel count after the final fusion projection.
		num_hyperedges : int = 8
			Number of adaptive hyperedges used in every `C3AH` block.
		num_heads : int = 8
			Number of heads used in every `AdaptiveHyperedgeGeneration` block.
		k : int = 2
			Number of parallel high-order `C3AH` branches.
		l : int = 1
			Number of repeated low-order `DS_C3k` blocks.
		c_h : float = 0.5
			Fraction of the fused mid-level width assigned to the high-order branch.
		c_l : float = 0.25
			Fraction of the fused mid-level width assigned to the low-order branch.
		c3ah_expansion : float = 1.0
			Expansion ratio used inside every `C3AH` block.
		low_order_depth : int = 1
			Inner `DS_Bottleneck` repeat count used by each low-order `DS_C3k` block.
		low_order_kernel : int = 3
			Kernel size used by each low-order `DS_Bottleneck` block.
		low_order_expansion : float = 1.0
			Expansion ratio used by each low-order `DS_C3k` block.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in all inner blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in the inner convolution blocks.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		conv_bias : bool = False
			Whether the convolution layers learn bias terms.
		linear_bias : bool = False
			Whether the linear layers inside the adaptive hypergraph blocks learn bias terms.

		Raises
		------
		ValueError
			Raised when the channel split implied by `c_h` and `c_l` leaves no positive shortcut
			width.
		"""
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
					self.c_h
					, self.c_h
					, num_hyperedges
					, num_heads
					, e=c3ah_expansion
					, activation=activation
					, norm_eps=norm_eps
					, norm_affine=norm_affine
					, conv_bias=conv_bias
					, linear_bias=linear_bias
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
					self.c_l
					, self.c_l
					, n=low_order_depth
					, k=low_order_kernel
					, e=low_order_expansion
					, activation=activation
					, norm_eps=norm_eps
					, norm_affine=norm_affine
					, bias=conv_bias
				)
				for _index in loops(l)
			]
		)

		self.final_fuse: Conv = Conv(
			self.c_h + self.c_l + self.c_s
			, out_channels
			, 1
			, 1
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, bias=conv_bias
		)

	def forward(self, x: list[Tensor]) -> Tensor:
		"""Return one HyperACE-conditioned fusion of four encoder stages.

		(AI generated docstring)

		You can use `forward` to resize four encoder-stage tensors to one shared mid-level spatial
		resolution, split the fused tensor into high-order, low-order, and shortcut branches, and fuse
		the branch outputs into one conditioning tensor for the decoder.

		Parameters
		----------
		x : list[Tensor]
			List of encoder tensors `[B2, B3, B4, B5]`.

		Returns
		-------
		conditioningTensor : Tensor
			Tensor of shape `(batch, out_channels, B4_height, B4_width)`.

		PyTorch
		-------
		stage alignment : `F.interpolate(..., mode='bilinear', align_corners=False)`
			`forward` resizes `B2`, `B3`, and `B5` to the spatial resolution of `B4` before channel
			concatenation [1]. The returned `conditioningTensor` therefore lives on the `B4` spatial
			grid.

		See Also
		--------
		C3AH
			Implement the high-order branch blocks.
		DS_C3k
			Implement the low-order branch blocks.
		HyperACE
			Store the branch partition and fusion layers used by `forward`.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
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
	"""Fuse one feature map with one resized HyperACE feature map.

	(AI generated docstring)

	You can use `GatedFusion` when one decoder feature map should absorb one resized HyperACE-guided
	conditioning feature map through a learnable residual gate [1].

	Mathematics
	-----------
	FullPAD gated fusion [1 at Equation (13)] : equation
	```
		Let Fᵢ ≜ `f_in`,  Hᵢ ≜ `h`,  Γ ≜ `self.gamma`,
			Γ ∈ ℝ^{1×C×1×1}

		F̃ᵢ = Fᵢ + Γ ⊙ Hᵢ

		where F̃ᵢ ≜ `fusedTensor`
	```

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(self, in_channels: int) -> None:
		"""Configure one gated-fusion layer.

		(AI generated docstring)

		You can use `__init__` to choose the channel count of the broadcastable gate parameter.

		Parameters
		----------
		in_channels : int
			Channel count shared by the decoder feature map and the resized HyperACE feature map.
		"""
		super().__init__()
		self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

	def forward(self, f_in: Tensor, h: Tensor) -> Tensor:
		"""Add one gated conditioning tensor to one decoder tensor.

		(AI generated docstring)

		You can use `forward` to inject one conditioning tensor `h` into one decoder feature tensor
		`f_in` through the broadcastable gate stored in `self.gamma`.

		Parameters
		----------
		f_in : Tensor
			Decoder feature tensor of shape `(batch, channels, height, width)`.
		h : Tensor
			Conditioning tensor of shape `(batch, channels, height, width)`.

		Returns
		-------
		fusedTensor : Tensor
			Tensor of shape `(batch, channels, height, width)`.

		Raises
		------
		ValueError
			Raised when `f_in.shape[1]` and `h.shape[1]` differ.

		PyTorch
		-------
		broadcast gate : parameter tensor of shape `(1, channels, 1, 1)`
			`forward` evaluates `f_in + self.gamma * h`, so `self.gamma` is broadcast across the batch
			axis and both spatial axes [1].

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		[2] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
			and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
			Visual Perception. https://arxiv.org/abs/2506.17733
		"""
		if f_in.shape[1] != h.shape[1]:
			message: str = (
				f"I received `{f_in.shape = }` and `{h.shape = }`, but I need the number of channels to match, so "
				f"`{f_in.shape[1] = }` to equal `{h.shape[1] = }`."
			)
			raise ValueError(message)
		return f_in + self.gamma * h

class Backbone(nn.Module):
	"""Encode one spectrogram-like feature map into four progressively coarser stages.

	(AI generated docstring)

	You can use `Backbone` to convert one `(channels, time, bands)` feature map into four encoder
	stages with progressively smaller spatial support and progressively larger channel width. The
	block uses the DS-series convolution blocks that HyperACE adopts from YOLOv13 [1].

	Mathematics
	-----------
	encoder recurrence : equation
	```
		Let X₀ ≜ `x`,  S ≜ `self.stem`,  P₂ ≜ `self.p2`,
			P₃ ≜ `self.p3`,  P₄ ≜ `self.p4`,  P₅ ≜ `self.p5`

		X₁ = S(X₀)
		B₂ = P₂(X₁)
		B₃ = P₃(B₂)
		B₄ = P₄(B₃)
		B₅ = P₅(B₄)

		where [B₂, B₃, B₄, B₅] ≜ `encoderStages`
	```

	See Also
	--------
	DSConv
		Implement the stride-carrying downsampling blocks.
	DS_C3k2
		Implement the per-stage refinement blocks.

	References
	----------
	[1] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(
		self
		, in_channels: int = 256
		, base_channels: int = 64
		, base_depth: int = 3
		, channels: tuple[int, int, int, int, int] | None = None
		, activation: type[nn.Module] = nn.SiLU
		, norm_eps: float = 1e-8
		, *
		, norm_affine: bool = True
		, conv_bias: bool = False
	) -> None:
		"""Configure one encoder backbone.

		(AI generated docstring)

		You can use `__init__` to choose the input width, the stage widths, the per-stage refinement
		depth, and the shared normalization or bias settings for one `Backbone` block.

		Parameters
		----------
		in_channels : int = 256
			Input channel count of the spectrogram feature map.
		base_channels : int = 64
			Stem output width used when `channels` is `None`.
		base_depth : int = 3
			Base refinement depth. The second and third encoder stages use `2 × base_depth`.
		channels : tuple[int, int, int, int, int] | None = None
			Explicit stage widths `(c2, c3, c4, c5, c6)`. When omitted, `Backbone` uses the default
			progression derived from `base_channels`.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in all inner blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in all inner convolution blocks.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		conv_bias : bool = False
			Whether the inner convolution layers learn bias terms.
		"""
		super().__init__()
		if channels is None:
			c2: int = base_channels
			# TODO Ought these values be user-configured, computed, or hardcoded?
			c3: int = 256
			c4: int = 384
			c5: int = 512
			c6: int = 768
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
		"""Return four progressively coarser encoder stages.

		(AI generated docstring)

		You can use `forward` to encode one band-split feature tensor `x` into the four stage tensors
		consumed later by `HyperACE` and `Decoder`.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, in_channels, time_frames, in_bands)`.

		Returns
		-------
		encoderStages : list[Tensor]
			List `[p2, p3, p4, p5]` of tensors in `NCHW` layout.

		PyTorch
		-------
		stage order : `p2`, `p3`, `p4`, `p5`
			`forward` returns the stage tensors in shallow-to-deep order so downstream code can unpack
			the list deterministically as `[p2, p3, p4, p5]`.

		See Also
		--------
		HyperACE
			Consume the returned stage list for multi-scale aggregation.
		Decoder
			Consume the returned stage list for coarse-to-fine decoding.

		References
		----------
		[1] `HyperACE`

		[2] `Decoder`
		"""
		x = self.stem(x)
		x2 = self.p2(x)
		x3 = self.p3(x2)
		x4 = self.p4(x3)
		x5 = self.p5(x4)
		return [x2, x3, x4, x5]

class Decoder(nn.Module):
	"""Fuse encoder stages with HyperACE features and decode them toward the input resolution.

	(AI generated docstring)

	You can use `Decoder` to combine the four encoder stages with one shared HyperACE feature map,
	decode those features from coarse to fine resolution, and return one refined feature map at the
	`p2` stage resolution. The block mixes DTTNet-style multi-stage decoding [1] with HyperACE's
	FullPAD-style gated feature injection [2].

	Mathematics
	-----------
	FullPAD projection [2 at Equation (12)] : equation
	```
		Let Y ≜ `h_ace`,  Rᵢ ≜ Resize(· → size(Fᵢ)),
			Pᵢ ∈ {`self.h_to_d5`, `self.h_to_d4`, `self.h_to_d3`, `self.h_to_d2`}

		Hᵢ = Pᵢ(Rᵢ(Y))
	```

	FullPAD gate [2 at Equation (13)] : equation
	```
		Let Γᵢ ∈ {`self.fusion_d5.gamma`, `self.fusion_d4.gamma`,
			`self.fusion_d3.gamma`, `self.fusion_d2.gamma`}

		F̃ᵢ = Fᵢ + Γᵢ ⊙ Hᵢ
	```

	coarse-to-fine decoder : recurrence
	```
		Let [P₂, P₃, P₄, P₅] ≜ `enc_feats`,
			Sᵢ ∈ {`self.skip_p5`, `self.skip_p4`, `self.skip_p3`, `self.skip_p2`},
			Uᵢ ∈ {`self.up_d5`, `self.up_d4`, `self.up_d3`},
			Gᵢ ∈ {`self.fusion_d5`, `self.fusion_d4`, `self.fusion_d3`, `self.fusion_d2`}

		D₅ = G₅(S₅(P₅), H₅)
		D₄ = G₄(U₅(Resize(D₅ → size(P₄))) + S₄(P₄), H₄)
		D₃ = G₃(U₄(Resize(D₄ → size(P₃))) + S₃(P₃), H₃)
		D₂ = G₂(U₃(Resize(D₃ → size(P₂))) + S₂(P₂), H₂)
		Y = `self.final_d2`(D₂)

		where Y ≜ `decodedTensor`
	```

	See Also
	--------
	GatedFusion
		Implement the per-stage FullPAD-style conditioning.
	HyperACE
		Produce the shared feature map injected into every decoder stage.

	References
	----------
	[1] Chen, J., Wu, L., Suryotrisongko, H., and Kim, M. (2024). Music source separation
		based on a lightweight deep learning framework (DTTNET: DUAL-PATH TFC-TDF UNET).
		https://arxiv.org/abs/2309.08684
	[2] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	"""

	def __init__(
		self
		, encoder_channels: list[int]
		, hyperace_out_c: int
		, decoder_channels: list[int]
		, block_depth: int = 1
		, block_kernel: int = 3
		, block_expansion: float = 0.5
		, activation: type[nn.Module] = nn.SiLU
		, norm_eps: float = 1e-8
		, *
		, conv_bias: bool = False
		, norm_affine: bool = True
	) -> None:
		"""Configure one coarse-to-fine decoder.

		(AI generated docstring)

		You can use `__init__` to choose the encoder widths, HyperACE conditioning width, decoder
		widths, inner DS-C3k refinement depth, and shared normalization or bias settings for one
		`Decoder` block.

		Parameters
		----------
		encoder_channels : list[int]
			Channel counts of the four encoder stages `[p2, p3, p4, p5]`.
		hyperace_out_c : int
			Channel count of the HyperACE conditioning feature map.
		decoder_channels : list[int]
			Decoder stage widths `[d2, d3, d4, d5]`.
		block_depth : int = 1
			Inner `DS_Bottleneck` repeat count used by each decoder `DS_C3k2` block.
		block_kernel : int = 3
			Kernel size used by each decoder `DS_Bottleneck` block.
		block_expansion : float = 0.5
			Expansion ratio used by each decoder `DS_C3k2` block.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in all inner blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in the inner convolution blocks.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		conv_bias : bool = False
			Whether the inner convolution layers learn bias terms.
		"""
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
			c_d5
			, c_d4
			, n=block_depth
			, k=block_kernel
			, e=block_expansion
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, bias=conv_bias
		)
		self.up_d4: DS_C3k2 = DS_C3k2(
			c_d4
			, c_d3
			, n=block_depth
			, k=block_kernel
			, e=block_expansion
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, bias=conv_bias
		)
		self.up_d3: DS_C3k2 = DS_C3k2(
			c_d3
			, c_d2
			, n=block_depth
			, k=block_kernel
			, e=block_expansion
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, bias=conv_bias
		)

		self.final_d2: DS_C3k2 = DS_C3k2(
			c_d2
			, c_d2
			, n=block_depth
			, k=block_kernel
			, e=block_expansion
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, bias=conv_bias
		)

	def forward(self, enc_feats: list[Tensor], h_ace: Tensor) -> Tensor:
		"""Decode four encoder stages with shared HyperACE conditioning.

		(AI generated docstring)

		You can use `forward` to combine the stage tensors in `enc_feats` with the conditioning tensor
		`h_ace`, decode the result from coarse to fine resolution, and return one refined tensor on
		the `p2` spatial grid.

		Parameters
		----------
		enc_feats : list[Tensor]
			List `[p2, p3, p4, p5]` of encoder-stage tensors.
		h_ace : Tensor
			Conditioning tensor produced by `HyperACE`.

		Returns
		-------
		decodedTensor : Tensor
			Tensor on the `p2` spatial grid with `decoder_channels[0]` channels.

		PyTorch
		-------
		resolution matching : bilinear interpolation toward skip-feature grids
			`forward` resizes coarse decoder states and the shared conditioning tensor to the spatial
			sizes of the skip tensors with `F.interpolate(..., mode='bilinear')` before each fusion
			step [1].

		See Also
		--------
		GatedFusion
			Inject the resized conditioning tensor at each decoder stage.
		HyperACE
			Produce the conditioning tensor consumed by `forward`.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
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
	"""Refine one feature map with residual TFC-TDF blocks.

	(AI generated docstring)

	You can use `TFC_TDF` to stack the TFC-TDF v3 residual block described in the DTTNet and
	TFC-TDF-UNet v3 literature [1][2]. Each repeated block applies one TFC path, one bottlenecked TDF
	path over the frequency axis, one second TFC path, and one residual shortcut.

	Mathematics
	-----------
	TFC-TDF residual stack : recurrence
	```
		Let X₀ ≜ `x`,  L ≜ |`self.blocks`|,
			Aₗ ≜ `self.blocks[ℓ].tfc1`,  Bₗ ≜ `self.blocks[ℓ].tdf`,
			Cₗ ≜ `self.blocks[ℓ].tfc2`,  Sₗ ≜ `self.blocks[ℓ].shortcut`

		Uₗ = Aₗ(Xₗ)
		Vₗ = Uₗ + Bₗ(Uₗ)
		Xₗ₊₁ = Cₗ(Vₗ) + Sₗ(Xₗ)
		Y = X_L

		where Y ≜ `refinedTensor`
	```

	Architecture
	------------
	per-block structure : TFC-TDF v3
		DTTNet describes one TFC-TDF v3 block as one TFC block, one residual TDF that compresses and
		restores the frequency axis, one second TFC block, and one residual convolution shortcut [1].

	References
	----------
	[1] Chen, J., Wu, L., Suryotrisongko, H., and Kim, M. (2024). Music source separation
		based on a lightweight deep learning framework (DTTNET: DUAL-PATH TFC-TDF UNET).
		https://arxiv.org/abs/2309.08684
	[2] Kim, M., Lee, J. H., and Jung, S. (2023). Sound Demixing Challenge 2023 Music
		Demixing Track Technical Report: TFC-TDF-UNet v3. https://arxiv.org/abs/2306.09382
	"""

	def __init__(self, in_c: int, c: int, l: int, f: int, bn: int = 4
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, norm_affine: bool = True
			, conv_bias: bool = False
			, linear_bias: bool = False) -> None:
		"""Configure one repeated TFC-TDF stack.

		(AI generated docstring)

		You can use `__init__` to choose the input width, hidden width, repeat count, frequency-axis
		width, TDF bottleneck factor, and normalization or bias settings for one `TFC_TDF` block.

		Parameters
		----------
		in_c : int
			Input channel count of the first repeated block.
		c : int
			Channel count used inside the repeated blocks and at the output.
		l : int
			Number of repeated TFC-TDF residual blocks.
		f : int
			Frequency-axis width expected by the TDF linear layers.
		bn : int = 4
			Bottleneck factor used to reduce the frequency axis from `f` to `f // bn` inside TDF.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in all inner blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in the inner normalization layers.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		conv_bias : bool = False
			Whether the convolution layers learn bias terms.
		linear_bias : bool = False
			Whether the TDF linear layers learn bias terms.
		"""
		super().__init__()

		self.blocks = nn.ModuleList()
		for _index in loops(l):
			block = nn.Module()

			block.tfc1 = nn.Sequential(
				nn.InstanceNorm2d(in_c, affine=norm_affine, eps=norm_eps), activation(), nn.Conv2d(in_c, c, 3, 1, 1, bias=conv_bias)
			)
			block.tdf = nn.Sequential(
				nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps)
				, activation()
				, nn.Linear(f, f // bn, bias=linear_bias)
				, nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps)
				, activation()
				, nn.Linear(f // bn, f, bias=linear_bias)
			)
			block.tfc2 = nn.Sequential(
				nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps), activation(), nn.Conv2d(c, c, 3, 1, 1, bias=conv_bias)
			)
			block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=conv_bias)

			self.blocks.append(block)
			in_c = c

	def forward(self, x: Tensor) -> Tensor:
		"""Refine one feature tensor with repeated TFC-TDF residual blocks.

		(AI generated docstring)

		You can use `forward` to pass one feature tensor `x` through each stored TFC-TDF residual
		block in sequence and return the final refined tensor.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, in_c_or_c, time_frames, frequency_bins)`.

		Returns
		-------
		refinedTensor : Tensor
			Tensor of shape `(batch, c, time_frames, frequency_bins)`.

		PyTorch
		-------
		frequency-axis linear layers : `nn.Linear` acts on the last dimension
			Each inner `tdf` stack applies `nn.Linear(f, f // bn)` and `nn.Linear(f // bn, f)`
			directly to one tensor in `NCHW` layout. In PyTorch, `nn.Linear` transforms the last
			dimension, so the TDF path operates along the frequency axis without permuting the tensor
			first [1].

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		[2] Chen, J., Wu, L., Suryotrisongko, H., and Kim, M. (2024). Music source separation
			based on a lightweight deep learning framework (DTTNET: DUAL-PATH TFC-TDF UNET).
			https://arxiv.org/abs/2309.08684
		"""
		for block in self.blocks:
			s: Tensor = cast('Callable[[Tensor], Tensor]', block.shortcut)(x)
			x = cast('Callable[[Tensor], Tensor]', block.tfc1)(x)
			x = x + cast('Callable[[Tensor], Tensor]', block.tdf)(x)
			x = cast('Callable[[Tensor], Tensor]', block.tfc2)(x)
			x = x + s
		return x

class FreqPixelShuffle(nn.Module):
	"""Expand frequency resolution with a channel-to-frequency rearrangement.

	(AI generated docstring)

	You can use `FreqPixelShuffle` to project one feature map to `out_channels × scale` channels,
	rearrange those channels into a wider frequency axis, and refine the result with one `TFC_TDF`
	block stack.

	Mathematics
	-----------
	channel-to-frequency rearrangement : transformation
	```
		Let X⁰ ≜ `x`,  C ≜ `self.conv`,  T ≜ `self.out_conv`,
			r ≜ `self.scale`,  X ≜ C(X⁰),  Cʳ ≜ |X|₁,  Cᵒ ≜ Cʳ / r

		Z[b, c, ρ, t, f] = X[b, cr + ρ, t, f]
		U[b, c, t, fr + ρ] = Z[b, c, ρ, t, f]
		Y = T(U)

		where Y ≜ `upsampledTensor`
	```

	See Also
	--------
	TFC_TDF
		Refine the frequency-expanded feature map.
	"""

	def __init__(
		self
		, in_channels: int
		, out_channels: int
		, scale: int
		, f: int
		, tfc_tdf_depth: int = 2
		, tfc_tdf_bn: int = 4
		, activation: type[nn.Module] = nn.SiLU
		, norm_eps: float = 1e-8
		, *
		, conv_bias: bool = False
		, linear_bias: bool = False
		, norm_affine: bool = True
	) -> None:
		"""Configure one frequency-axis pixel-shuffle stage.

		(AI generated docstring)

		You can use `__init__` to choose the input width, output width, frequency upsampling factor,
		post-shuffle frequency width, TFC-TDF depth, and normalization or bias settings for one
		`FreqPixelShuffle` block.

		Parameters
		----------
		in_channels : int
			Input channel count.
		out_channels : int
			Output channel count after channel-to-frequency rearrangement.
		scale : int
			Frequency upsampling factor.
		f : int
			Frequency-axis width after rearrangement. `FreqPixelShuffle` forwards `f` to `TFC_TDF`.
		tfc_tdf_depth : int = 2
			Number of repeated TFC-TDF blocks used after rearrangement.
		tfc_tdf_bn : int = 4
			TDF bottleneck factor forwarded to `TFC_TDF`.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in the inner `DSConv` and `TFC_TDF` blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in the inner blocks.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		conv_bias : bool = False
			Whether the convolution layers learn bias terms.
		linear_bias : bool = False
			Whether the TDF linear layers learn bias terms.
		"""
		super().__init__()
		self.scale: int = scale
		self.conv: DSConv = DSConv(
			in_channels, out_channels * scale, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)
		self.out_conv: TFC_TDF = TFC_TDF(
			out_channels
			, out_channels
			, tfc_tdf_depth
			, f
			, bn=tfc_tdf_bn
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
			, linear_bias=linear_bias
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Increase frequency resolution by rearranging channels into bins.

		(AI generated docstring)

		You can use `forward` to expand the last spatial axis of `x` by `self.scale`, reinterpret
		projected channels as frequency positions, and refine the expanded tensor with the stored
		`TFC_TDF` block.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, in_channels, time_frames, frequency_bins)`.

		Returns
		-------
		upsampledTensor : Tensor
			Tensor of shape `(batch, out_channels, time_frames, frequency_bins × scale)`.

		PyTorch
		-------
		channel-to-frequency rearrangement : `view`, `permute`, `contiguous`, `view`
			`forward` first projects `x` to `out_channels × scale` channels, then reshapes and
			permutes the tensor so that the extra channel factor becomes additional positions on the
			last spatial axis. `contiguous()` makes the memory layout safe for the final `view` [1].

		See Also
		--------
		TFC_TDF
			Refine the frequency-expanded tensor.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		x = self.conv(x)
		B, C_r, H, W = x.shape
		out_c: int = C_r // self.scale

		x = x.view(B, out_c, self.scale, H, W)

		x = x.permute(0, 1, 3, 4, 2).contiguous()
		x = x.view(B, out_c, H, W * self.scale)

		return self.out_conv(x)

class ProgressiveUpsampleHead(nn.Module):
	"""Recover target frequency resolution through repeated frequency-axis upsampling.

	(AI generated docstring)

	You can use `ProgressiveUpsampleHead` to convert the decoder output at the band-split frequency
	resolution into one mask tensor defined on the requested number of STFT bins. The head applies
	four `FreqPixelShuffle` stages and one final `3 × 3` projection.

	Mathematics
	-----------
	progressive frequency expansion : recurrence
	```
		Let X₀ ≜ `x`,  C ≜ `self.final_conv`,  T ≜ `self.target_bins`,
			Bᵢ ∈ {`self.block1`, `self.block2`, `self.block3`, `self.block4`}

		X₁ = B₁(X₀)
		X₂ = B₂(X₁)
		X₃ = B₃(X₂)
		X₄ = B₄(X₃)
		Z = ⎧ Resize(X₄ → (|X₄|₂, T)),  |X₄|₃ ≠ T
			⎨
			⎩ X₄,                       |X₄|₃ = T
		Y = C(Z)

		where Y ≜ `maskTensor`
	```

	See Also
	--------
	FreqPixelShuffle
		Implement each progressive frequency upsampling stage.
	"""

	def __init__(
		self
		, in_channels: int
		, out_channels: int
		, target_bins: int = 1025
		, in_bands: int = 62
		, upsample_scales: tuple[int, int, int, int] = (2, 2, 2, 2)
		, tfc_tdf_depth: int = 2
		, tfc_tdf_bn: int = 4
		, activation: type[nn.Module] = nn.SiLU
		, norm_eps: float = 1e-8
		, *
		, conv_bias: bool = False
		, linear_bias: bool = False
		, norm_affine: bool = True
	) -> None:
		"""Configure one progressive frequency upsampling head.

		(AI generated docstring)

		You can use `__init__` to choose the decoder width, output mask width, target STFT-bin count,
		frequency upsampling schedule, and shared TFC-TDF settings for one `ProgressiveUpsampleHead`.

		Parameters
		----------
		in_channels : int
			Input channel count from the decoder.
		out_channels : int
			Output channel count of the final mask tensor.
		target_bins : int = 1025
			Target number of STFT bins at the output.
		in_bands : int = 62
			Frequency-bin count of the decoder input before progressive upsampling.
		upsample_scales : tuple[int, int, int, int] = (2, 2, 2, 2)
			Per-stage frequency upsampling factors used by the four `FreqPixelShuffle` blocks.
		tfc_tdf_depth : int = 2
			Number of repeated TFC-TDF blocks used inside every `FreqPixelShuffle` block.
		tfc_tdf_bn : int = 4
			TDF bottleneck factor used inside every `FreqPixelShuffle` block.
		activation : type[nn.Module] = nn.SiLU
			Activation class instantiated in all inner blocks.
		norm_eps : float = 1e-8
			Instance-normalization epsilon used in all inner blocks.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters.
		conv_bias : bool = False
			Whether the convolution layers learn bias terms.
		linear_bias : bool = False
			Whether the TDF linear layers learn bias terms.
		"""
		super().__init__()
		self.target_bins: int = target_bins

		c: int = in_channels
		scale1, scale2, scale3, scale4 = upsample_scales
		f1: int = in_bands * scale1
		f2: int = f1 * scale2
		f3: int = f2 * scale3
		f4: int = f3 * scale4

		self.block1: FreqPixelShuffle = FreqPixelShuffle(
			c
			, c // 2
			, scale=scale1
			, f=f1
			, tfc_tdf_depth=tfc_tdf_depth
			, tfc_tdf_bn=tfc_tdf_bn
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
			, linear_bias=linear_bias
		)
		self.block2: FreqPixelShuffle = FreqPixelShuffle(
			c // 2
			, c // 4
			, scale=scale2
			, f=f2
			, tfc_tdf_depth=tfc_tdf_depth
			, tfc_tdf_bn=tfc_tdf_bn
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
			, linear_bias=linear_bias
		)
		self.block3: FreqPixelShuffle = FreqPixelShuffle(
			c // 4
			, c // 8
			, scale=scale3
			, f=f3
			, tfc_tdf_depth=tfc_tdf_depth
			, tfc_tdf_bn=tfc_tdf_bn
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
			, linear_bias=linear_bias
		)
		self.block4: FreqPixelShuffle = FreqPixelShuffle(
			c // 8
			, c // 16
			, scale=scale4
			, f=f4
			, tfc_tdf_depth=tfc_tdf_depth
			, tfc_tdf_bn=tfc_tdf_bn
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
			, linear_bias=linear_bias
		)

		self.final_conv = nn.Conv2d(c // 16, out_channels, kernel_size=3, stride=1, padding='same', bias=conv_bias)

	def forward(self, x: Tensor) -> Tensor:
		"""Return one full-resolution mask tensor from decoder features.

		(AI generated docstring)

		You can use `forward` to increase the frequency resolution of decoder tensor `x` through the
		four stored `FreqPixelShuffle` stages and then project the result to the final mask channels.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, in_channels, time_frames, in_bands)`.

		Returns
		-------
		maskTensor : Tensor
			Tensor of shape `(batch, out_channels, time_frames, target_bins)`.

		PyTorch
		-------
		final bin alignment : exact last-axis correction with `F.interpolate`
			Each `FreqPixelShuffle` block increases only the last spatial axis. If the accumulated
			scale factors do not land exactly on `self.target_bins`, `forward` uses `F.interpolate`
			with `size=(x.shape[2], self.target_bins)` to preserve the time axis and correct only the
			frequency axis [1].

		See Also
		--------
		FreqPixelShuffle
			Implement each progressive frequency-axis upsampling stage.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		if x.shape[-1] != self.target_bins:
			x = F.interpolate(x, size=(x.shape[2], self.target_bins), mode='bilinear', align_corners=False)

		return self.final_conv(x)

class SegmModel(nn.Module):
	"""Assemble the full segmentation-style HyperACE branch used by the mask estimator.

	You can use `SegmModel` to encode one band-split feature tensor, aggregate the encoder stages with
	`HyperACE`, decode the aggregated representation, restore the time resolution, and upsample the
	frequency axis to the target STFT-bin count. `bandSplit.MaskEstimator` uses this branch when
	`use_hyperACE` is enabled [1].

	Mathematics
	-----------
	HyperACE mask branch : equation
	```
		Let X ≜ `x`,  B ≜ `self.backbone`,  H ≜ `self.hyperace`,
			D ≜ `self.decoder`,  U ≜ `self.upsample_head`,  T₀ ≜ |X|₂

		E = B(X)
		A = H(E)
		Z = D(E, A)
		Z′ = Resize(Z → (T₀, |Z|₃))
		Y = U(Z′)

		where Y ≜ `mask_tensor`
	```

	See Also
	--------
	Backbone
		Encode the band-split feature tensor into four stages.
	HyperACE
		Aggregate those encoder stages with adaptive hypergraph modeling.
	Decoder
		Decode the encoder stages with HyperACE conditioning.
	ProgressiveUpsampleHead
		Restore the target frequency resolution and emit output masks.

	References
	----------
	[1] hunterFormsBS.bandSplit.MaskEstimator

	[2] Lei, M., Li, S., Wu, Y., Hu, H., Zhou, Y., Zheng, X., Ding, G., Du, S., Wu, Z.,
		and Gao, Y. (2025). YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive
		Visual Perception. https://arxiv.org/abs/2506.17733
	[3] Lu, W.-T., Wang, J.-C., Kong, Q., and Hung, Y.-N. (2023). Music Source Separation
		with Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
	[4] Wang, J.-C., Lu, W.-T., and Chen, J. (2024). Mel-RoFormer for Vocal Separation and
		Vocal Melody Transcription. https://arxiv.org/abs/2409.04702
	"""

	def __init__(
		self
		, activation: type[nn.Module] = nn.SiLU
		, backbone_channels: tuple[int, int, int, int, int] | None = None
		, base_channels: int = 64
		, base_depth: int = 2
		, decoder_block_depth: int = 1
		, decoder_block_expansion: float = 0.5
		, decoder_block_kernel: int = 3
		, decoder_channels: list[int] | tuple[int, int, int, int] | None = None
		, hyperace_c_h: float = 0.5
		, hyperace_c_l: float = 0.25
		, hyperace_c3ah_expansion: float = 1.0
		, hyperace_k: int = 2
		, hyperace_l: int = 1
		, hyperace_low_order_depth: int = 1
		, hyperace_low_order_expansion: float = 1.0
		, hyperace_low_order_kernel: int = 3
		, hyperace_out_channels: int | None = None
		, in_bands: int = 62
		, in_dim: int = 256
		, norm_eps: float = 1e-8
		, num_heads: int = 8
		, num_hyperedges: int = 32
		, out_bins: int = 1025
		, out_channels: int = 4
		, upsample_scales: tuple[int, int, int, int] = (2, 2, 2, 2)
		, upsample_tfc_tdf_bn: int = 4
		, upsample_tfc_tdf_depth: int = 2
		, *
		, conv_bias: bool = False
		, linear_bias: bool = False
		, norm_affine: bool = True
	) -> None:
		"""Configure one full HyperACE segmentation branch.

		(AI generated docstring)

		You can use `__init__` to choose the input band geometry, the encoder width schedule, the
		adaptive-hypergraph size, the decoder width schedule, and the final frequency upsampling plan
		for one `SegmModel` instance.

		Parameters
		----------
		in_bands : int = 62
			Input number of band-split frequency bins presented to `SegmModel`.
		in_dim : int = 256
			Input channel count of the band-split feature tensor.
		out_bins : int = 1025
			Target number of STFT bins produced by `ProgressiveUpsampleHead`.
		out_channels : int = 4
			Output channel count of the final mask tensor.
		activation : type[nn.Module] = nn.SiLU
			Activation class shared by submodules that expose `activation`.
		norm_eps : float = 1e-8
			Instance-normalization epsilon shared by submodules that expose `norm_eps`.
		norm_affine : bool = True
			Whether instance normalization learns affine parameters in submodules that expose
			`norm_affine`.
		conv_bias : bool = False
			Whether convolution layers learn bias terms in submodules that expose `conv_bias`.
		linear_bias : bool = False
			Whether linear layers learn bias terms in submodules that expose `linear_bias`.

		Other Parameters
		----------------
		backbone parameters : forwarded parameter family
			See `Backbone.__init__`. `SegmModel` forwards `base_channels`, `base_depth`, and
			`backbone_channels` to `Backbone`.
		hypergraph size : forwarded parameter family
			See `HyperACE.__init__`. `SegmModel` forwards `num_hyperedges` and `num_heads` to
			`HyperACE`.
		hyperace_* : forwarded parameter family
			See `HyperACE.__init__`.
		decoder_channels : forwarded parameter
			See `Decoder.__init__`. When `decoder_channels` is `None`, `SegmModel` mirrors the encoder
			widths `[c2, c3, c4, c5]` before calling `Decoder`.
		decoder_* : forwarded parameter family
			See `Decoder.__init__`.
		upsample_* : forwarded parameter family
			See `ProgressiveUpsampleHead.__init__`.
		"""
		super().__init__()

		self.backbone = Backbone(
			in_channels=in_dim
			, base_channels=base_channels
			, base_depth=base_depth
			, channels=backbone_channels
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
		)
		enc_channels: list[int] = self.backbone.out_channels
		c2, c3, c4, c5 = enc_channels

		hyperace_in_channels: list[int] = enc_channels
		hyperace_out_channels = c4 if hyperace_out_channels is None else hyperace_out_channels
		self.hyperace = HyperACE(
			hyperace_in_channels
			, hyperace_out_channels
			, num_hyperedges
			, num_heads
			, k=hyperace_k
			, l=hyperace_l
			, c_h=hyperace_c_h
			, c_l=hyperace_c_l
			, c3ah_expansion=hyperace_c3ah_expansion
			, low_order_depth=hyperace_low_order_depth
			, low_order_kernel=hyperace_low_order_kernel
			, low_order_expansion=hyperace_low_order_expansion
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
			, linear_bias=linear_bias
		)

		decoder_channels = list(decoder_channels or (c2, c3, c4, c5))
		self.decoder = Decoder(
			enc_channels
			, hyperace_out_channels
			, decoder_channels
			, block_depth=decoder_block_depth
			, block_kernel=decoder_block_kernel
			, block_expansion=decoder_block_expansion
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
		)

		self.upsample_head = ProgressiveUpsampleHead(
			in_channels=decoder_channels[0]
			, out_channels=out_channels
			, target_bins=out_bins
			, in_bands=in_bands
			, upsample_scales=upsample_scales
			, tfc_tdf_depth=upsample_tfc_tdf_depth
			, tfc_tdf_bn=upsample_tfc_tdf_bn
			, activation=activation
			, norm_eps=norm_eps
			, norm_affine=norm_affine
			, conv_bias=conv_bias
			, linear_bias=linear_bias
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Predict full-resolution masks from one band-split feature tensor.

		(AI generated docstring)

		You can use `forward` to encode one band-split feature tensor `x`, aggregate multi-scale
		correlations with `HyperACE`, decode the aggregated tensor back to finer time resolution, and
		emit one full-resolution mask tensor.

		Parameters
		----------
		x : Tensor
			Input tensor of shape `(batch, in_dim, time_frames, in_bands)`.

		Returns
		-------
		mask_tensor : Tensor
			Tensor of shape `(batch, out_channels, time_frames, out_bins)`.

		PyTorch
		-------
		time-axis restoration : bilinear interpolation before frequency upsampling
			`Backbone` reduces spatial support before `Decoder` reconstructs one tensor on the `p2`
			grid. `forward` then uses `F.interpolate` with `size=(H, dec_feat.shape[-1])` to restore
			the original time-axis length before `ProgressiveUpsampleHead` expands the frequency axis
			[1].

		See Also
		--------
		Backbone
			Encode the input tensor into four stage tensors.
		HyperACE
			Aggregate the stage tensors into one conditioning tensor.
		Decoder
			Decode the stage tensors with HyperACE conditioning.
		ProgressiveUpsampleHead
			Expand the frequency axis to `out_bins` and emit the final mask channels.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		[2] `Backbone`

		[3] `HyperACE`

		[4] `Decoder`

		[5] `ProgressiveUpsampleHead`
		"""
		H, _W = x.shape[2:]

		enc_feats = self.backbone(x)

		h_ace_feats = self.hyperace(enc_feats)

		dec_feat = self.decoder(enc_feats, h_ace_feats)

		feat_time_restored: Tensor = F.interpolate(dec_feat, size=(H, dec_feat.shape[-1]), mode='bilinear', align_corners=False)

		return self.upsample_head(feat_time_restored)
