# ruff: noqa: PLC0415
"""Assemble attention, feedforward, and transformer blocks for music source separation.

You can use this module to assemble the unified attention stack shared by `BandSplitRotator`,
`BSRoformer`, and `MelBandRoformer` [1][2][3]. The constructor parameter names stay aligned across the
stack so wrapper classes can pass settings such as `attn_dropout`, `flash_attn`, `sage_attention`,
`scale`, and value-residual-learning switches downstream without renaming [1][2][3]. `Attend`
evaluates attention from precomputed query, key, and value arrays and can dispatch to an explicit
implementation, to PyTorch SDPA [4], or to SageAttention [5][6]. `Attention` projects activations into
query, key, and value arrays, applies RoPE [7] or PoPE [8], optionally mixes a learned value residual,
and then calls `Attend`. `FeedForward` applies the position-wise expansion-and-projection block paired
with each attention block. `Transformer` stacks repeated `Attention` and `FeedForward` pairs and
optionally installs multi-stream residual adapters for the wrapper models [1][2][3].

Contents
--------
Classes
	Attend
		Evaluate attention output from precomputed query, key, and value arrays.
	Attention
		Project activations into multi-head query, key, and value arrays and return gated output.
	FeedForward
		Apply a position-wise expansion-and-projection nonlinear block.
	Transformer
		Stack repeated `Attention` and `FeedForward` pairs into a residual sequence.

References
----------
[1] hunterFormsBS.bandSplitRotator.BandSplitRotator

[2] hunterFormsBS.bs_roformer.BSRoformer

[3] hunterFormsBS.mel_band_roformer.MelBandRoformer

[4] PyTorch.
	https://context7.com/pytorch/pytorch
[5] Zhang, J., Wei, J., Huang, H., Zhang, P., Zhu, J., and Chen, J. (2025).
	SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.
	https://arxiv.org/abs/2410.02367
[6] thu-ml/SageAttention
	https://github.com/thu-ml/SageAttention
[7] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
	RoFormer: Enhanced Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
[8] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
	Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings.
	https://arxiv.org/abs/2509.10534
"""
from __future__ import annotations

from einops import rearrange
from hunterFormsBS.theTypes import FlashAttentionConfig, ParametersAttention
from hunterMakesPy import raiseIfNone
from hyper_connections import get_init_and_expand_reduce_stream_functions  # NOTE There is a newer version.
from more_itertools import loops
from operator import neg
from packaging import version
from PoPE_pytorch import flash_attn_with_pope, PoPE
from torch import einsum, nn, Tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import default, exists, once
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast, overload, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
	from rotary_embedding_torch import RotaryEmbedding
	from torch._C import _CudaDeviceProperties

print_once = once(print)

class Attend(nn.Module):
	"""Compute attention output from precomputed query, key, and value arrays.

	You can use `Attend` after `Attention` [1] has already built query `Tensor` `q`, key `Tensor` `k`,
	and value `Tensor` `v`. `Attend` compares query `Tensor` `q` against key `Tensor` `k` to build
	attention weights and then uses the attention weights to mix value `Tensor` `v` with the standard
	scaled dot-product rule [2]. `Attend` can compute that mapping with an explicit implementation,
	with PyTorch SDPA [3], or with SageAttention [4][5] when the optional dependency is installed
	manually.

	Attributes
	----------
	nn_Dropout : nn.Dropout
		Dropout module applied to attention weights after `softmax` in the explicit path.
	attn_dropout : float
		Probability used for attention-weight dropout during training.
	cpu_config : FlashAttentionConfig
		Backend flags passed to `torch.backends.cuda.sdp_kernel` [3] when execution remains on CPU or
		when `cuda_config` is `None` [6].
	cuda_config : FlashAttentionConfig | None
		Backend flags used for CUDA execution when `flash` is enabled [6].
	flash : bool
		Whether `forward` may delegate to `flash_attn` when `sage_attention` is `False`.
	sage_attention : bool
		Whether `forward` should try `sageattention.sageattn` [4][5] before the PyTorch SDPA path or
		the explicit path.
	scale : float = q.shape[-1] ** -0.5
		Attention-score multiplier used in every backend.

	See Also
	--------
	Attention
		Project model activations into query, key, and value arrays for multi-head attention.

	References
	----------
	[1] hunterFormsBS.attend.Attention

	[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
		Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need.
		https://arxiv.org/abs/1706.03762
	[3] PyTorch.
		https://context7.com/pytorch/pytorch
	[4] Zhang, J., Wei, J., Huang, H., Zhang, P., Zhu, J., and Chen, J. (2025).
		SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.
		https://arxiv.org/abs/2410.02367
	[5] thu-ml/SageAttention
		https://github.com/thu-ml/SageAttention
	[6] hunterFormsBS.theTypes.FlashAttentionConfig
	"""

	def __init__(self, attn_dropout: float, scale: float, *, flash: bool = False, sage_attention: bool = False) -> None:
		"""Configure attention-score scaling and backend selection.

		You can use `__init__` to set the attention-weight dropout probability, store the
		attention-score multiplier, and select which fallback backends `forward` may use. `__init__`
		prepares one PyTorch SDPA configuration for CPU execution and, when `flash=True` and a
		supported CUDA device is available, one PyTorch SDPA configuration for CUDA execution [1].
		`__init__` also stores whether `forward` should prefer SageAttention [2][3], which you must
		install manually outside this package.

		Parameters
		----------
		attn_dropout : float
			Probability applied to attention weights during training.
		scale : float
			Attention-score multiplier used by every backend.
		flash : bool = False
			Whether `forward` may delegate to PyTorch SDPA [1] through `flash_attn` when
			`sage_attention` is `False`.
		sage_attention : bool = False
			Whether `forward` should first try `sageattention.sageattn` [2][3]. `hunterFormsBS` does
			not install `sageattention`.

		PyTorch
		-------
		backend selection : implementation detail
			`__init__` inspects the active CUDA device when `flash` is `True`. `__init__` enables the
			flash backend only on A100 devices and keeps the math plus memory-efficient backends
			available on other CUDA devices [1].

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		[2] Zhang, J., Wei, J., Huang, H., Zhang, P., Zhu, J., and Chen, J. (2025).
			SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.
			https://arxiv.org/abs/2410.02367
		[3] thu-ml/SageAttention
			https://github.com/thu-ml/SageAttention
		"""
		super().__init__()
		self.scale: float = scale
		self.attn_dropout: float = attn_dropout
		self.nn_Dropout: nn.Dropout = nn.Dropout(self.attn_dropout)

		self.flash: bool = flash
		if self.flash and version.parse(torch.__version__) < version.parse('2.0.0'):
			pytorchVersion: str = torch.__version__
			message: str = f'I received `{pytorchVersion = }`, but `flash=True` requires PyTorch 2.0.0 or above.'
			raise RuntimeError(message)

		self.sage_attention: bool = sage_attention

		# determine efficient attention configs for cuda and cpu

		self.cpu_config: FlashAttentionConfig = FlashAttentionConfig(enable_flash=True, enable_math=True, enable_mem_efficient=True)
		self.cuda_config: FlashAttentionConfig | None = None

		if not torch.cuda.is_available() or not self.flash:
			return

		device_properties = torch.cuda.get_device_properties(torch.device('cuda')) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
		device_properties = cast('_CudaDeviceProperties', device_properties)

		if device_properties.major == 8 and device_properties.minor == 0:
			print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
			self.cuda_config = FlashAttentionConfig(enable_flash=True, enable_math=False, enable_mem_efficient=False)
		else:
			print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
			self.cuda_config = FlashAttentionConfig(enable_flash=False, enable_math=True, enable_mem_efficient=True)

	def flash_attn(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
		"""Compute attention output from query, key, and value arrays with PyTorch SDPA.

		You can use `flash_attn` when query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v`
		already share compatible batch, head, sequence, and head-feature axes. `flash_attn` is the
		PyTorch-SDPA path selected by `forward` [1] when `sage_attention` is `False` and `flash` is
		`True`. `flash_attn` rescales query `Tensor` `q` when `self.scale` differs from the default
		inverse-square-root query-feature normalization [2]. `flash_attn` then selects a PyTorch SDPA
		backend with `torch.backends.cuda.sdp_kernel` [3] and returns the attention-weighted mixture
		of value `Tensor` `v` through `torch.nn.functional.scaled_dot_product_attention` [3]. When the
		selected backend supports it, PyTorch may realize the computation with a flash-style kernel
		[4].

		Parameters
		----------
		q : Tensor
			Query `Tensor` with shape `batch × head × query position × head feature`.
		k : Tensor
			Key `Tensor` with shape `batch × head × key position × head feature`.
		v : Tensor
			Value `Tensor` with shape `batch × head × key position × head feature`.

		Returns
		-------
		out : Tensor
			Attention output `Tensor` with shape `batch × head × query position × head feature`.

		Mathematics
		-----------
		scaled scores : equation
		```
			Let d ≜ `q.shape[-1]`,  s₀ ≜ `self.scale`,  Q ≜ `q`,  K ≜ `k`,  V ≜ `v`,
				s₀ ∈ ℝ ∪ {∅}

				s = ⎧ s₀,     s₀ ≠ ∅
					⎨
					⎩ d⁻¹ᐟ²,  s₀ = ∅

			Q′ = (s / d⁻¹ᐟ²) Q
			S[b, h, i, j] = ⟨Q′[b, h, i, :], K[b, h, j, :]⟩ · d⁻¹ᐟ² = ⟨Q[b, h, i, :], K[b, h, j, :]⟩ · s
			A[b, h, i, j] = exp(S[b, h, i, j]) / ∑ₘ exp(S[b, h, i, m])
			O[b, h, i, :] = ∑ⱼ A[b, h, i, j] · V[b, h, j, :]

			where  O ≜ `out`
		```

		PyTorch
		-------
		backend selection : implementation detail
			`flash_attn` always enters `torch.backends.cuda.sdp_kernel` [1] before calling
			`torch.nn.functional.scaled_dot_product_attention` [1]. `cpu_config` enables all three
			SDPA backend families. When `q.is_cuda` is `True` and `cuda_config` is available,
			`flash_attn` switches to the stored CUDA-specific flags. During training, `flash_attn`
			passes `self.attn_dropout`, and during evaluation, `flash_attn` passes `0.0`.

		References
		----------
		[1] hunterFormsBS.attend.Attend.forward

		[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
			Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need.
			https://arxiv.org/abs/1706.03762
		[3] PyTorch.
			https://context7.com/pytorch/pytorch
		[4] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022).
			FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
			https://arxiv.org/abs/2205.14135
		"""
		is_cuda: bool = q.is_cuda

		q = q * (self.scale / (q.shape[-1] ** neg(0.5)))

		# Check if there is a compatible device for flash attention
		config: FlashAttentionConfig = self.cpu_config
		if is_cuda and exists(self.cuda_config):
			config = self.cuda_config

		# pytorch 2.0 flash attn: q, k, v, mask, attn_dropout, softmax_scale
		with torch.backends.cuda.sdp_kernel(**config._asdict()): # pyright: ignore[reportDeprecated]
			out: Tensor = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)

		return out

	def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
		"""Compute attention output from query, key, and value arrays.

		You can use `forward` after another module has already built query `Tensor` `q`, key `Tensor`
		`k`, and value `Tensor` `v`. `forward` compares query `Tensor` `q` against key `Tensor` `k` to
		build attention weights and then uses those attention weights to mix value `Tensor` `v`. When
		`sage_attention` is `True`, `forward` imports and calls `sageattention.sageattn(q, k, v,
		tensor_layout='HND', is_causal=False)` [1][2]. When `sage_attention` is `False` and `flash` is
		`True`, `forward` delegates to `flash_attn` [3]. Otherwise `forward` computes the standard
		scaled dot-product attention weights explicitly with `softmax` [4].

		Parameters
		----------
		q : Tensor
			Query `Tensor` with shape `batch × head × query position × head feature`.
		k : Tensor
			Key `Tensor` with shape `batch × head × key position × head feature`.
		v : Tensor
			Value `Tensor` with shape `batch × head × key position × head feature`.

		Returns
		-------
		out : Tensor
			Attention output `Tensor` with shape `batch × head × query position × head feature`.

		SageAttention
		-------------
		manual installation : dependency
			`hunterFormsBS` does not install `sageattention`. Install `sageattention` manually from
			the SageAttention project [2] before enabling `sage_attention`.

		Mathematics
		-----------
		scaled scores : equation
		```
			Let d ≜ `q.shape[-1]`,  s₀ ≜ `self.scale`,  Q ≜ `q`,  K ≜ `k`,  V ≜ `v`,
				s₀ ∈ ℝ ∪ {∅}

				s = ⎧ s₀,     s₀ ≠ ∅
					⎨
					⎩ d⁻¹ᐟ²,  s₀ = ∅

			S[b, h, i, j] = ⟨Q[b, h, i, :], K[b, h, j, :]⟩ · s
			A[b, h, i, j] = exp(S[b, h, i, j]) / ∑ₘ exp(S[b, h, i, m])
			O[b, h, i, :] = ∑ⱼ A[b, h, i, j] · V[b, h, j, :]

			where  S ≜ `similarity`,  A ≜ `attention_weights`
		```

		References
		----------
		[1] Zhang, J., Wei, J., Huang, H., Zhang, P., Zhu, J., and Chen, J. (2025).
			SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.
			https://arxiv.org/abs/2410.02367
		[2] thu-ml/SageAttention
			https://github.com/thu-ml/SageAttention
		[3] hunterFormsBS.attend.Attend.flash_attn

		[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
			Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need.
			https://arxiv.org/abs/1706.03762
		"""
		if self.sage_attention:
			from sageattention import sageattn  # pyright: ignore[reportMissingImports, reportUnknownVariableType] # ty:ignore[unresolved-import]
			return sageattn(q, k, v, tensor_layout='HND', is_causal=False) # pyright: ignore[reportUnknownVariableType]
		if self.flash:
			return self.flash_attn(q, k, v)

		similarity: Tensor = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

		attention_weights: Tensor = similarity.softmax(dim=-1)
		attention_weights = self.nn_Dropout(attention_weights)

		# aggregate values

		return einsum('b h i j, b h j d -> b h i d', attention_weights, v)

class Attention(nn.Module):
	"""Mix information across sequence positions and return updated features.

	You can use `Attention` when activation `Tensor` `x` at each sequence position should gather
	information from other sequence positions before the next block. `Attention` projects activation
	`Tensor` `x` into query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v`, applies PoPE
	[1][2] or RoPE [3] when those position encoders are available, and aggregates the result through
	`Attend` [4]. `Attention` can also mix an incoming value residual into value `Tensor` `v` before
	attention and can return the unmixed value `Tensor` so outer stacks such as `Transformer` [5],
	`BandSplitRotator` [6], `BSRoformer` [7], and `MelBandRoformer` [8] can thread the value residual
	through multiple blocks.

	PyTorch
	-------
	module structure : implementation detail
		`Attention` applies `RMSNorm` to the input activations, projects the normalized activations
		with `to_qkv`, reshapes the projected result into head-specific query `Tensor` `q`, key
		`Tensor` `k`, and value `Tensor` `v`, and then chooses one of three execution paths. When
		`self.learned_value_residual_mix` is not `None` and `value_residual` is provided, `Attention`
		computes a sigmoid mixing factor and linearly interpolates `v` toward `value_residual`.
		`Attention` calls `flash_attn_with_pope` [2] when `pope_embed` is available, rotates query
		`Tensor` `q` and key `Tensor` `k` before `Attend` [4] when `rotary_embed` is available, and
		otherwise delegates directly to `Attend` [4]. After aggregation, `Attention` multiplies each
		head output by a sigmoid gate from `to_gates`, concatenates the gated head outputs, applies
		`to_out`, and optionally returns the pre-mixed value `Tensor`.

	Attributes
	----------
	attend : Attend
		Exact attention core used when `pope_embed` is `None` [4].
	heads : int
		Number of attention heads.
	learned_value_residual_mix : nn.Linear | None
		Optional projection that computes one value-residual mixing factor per head.
	norm : RMSNorm
		Root-mean-square normalization module applied before the projections.
	pope_embed : PoPE | None
		Optional polar-coordinate position encoder [1][2].
	rotary_embed : RotaryEmbedding | None
		Optional rotary position encoder [3].
	scale : float = dim_head ** -0.5
		Head-feature normalization factor. `scale` is `dim_head ** -0.5` when `scale` is omitted.
	to_gates : nn.Linear
		Projection layer implemented with `nn.Linear` [9] to produce one gate value for each head.
	to_out : nn.Sequential
		Output projection and dropout submodule sequence implemented with PyTorch layers [9].
	to_qkv : nn.Linear
		Projection layer implemented with `nn.Linear` [9] to produce concatenated query, key, and
		value features.

	See Also
	--------
	Attend
		Aggregate precomputed query, key, and value arrays.
	FeedForward
		Apply the position-wise sublayer that follows `Attention` inside `Transformer`.
	Transformer
		Stack `Attention` and `FeedForward` blocks with residual logic.

	References
	----------
	[1] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
		Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings.
		https://arxiv.org/abs/2509.10534
	[2] lucidrains/PoPE-pytorch.
		https://github.com/lucidrains/PoPE-pytorch
	[3] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
		RoFormer: Enhanced Transformer with Rotary Position Embedding.
		https://arxiv.org/abs/2104.09864
	[4] hunterFormsBS.attend.Attend

	[5] hunterFormsBS.attend.Transformer

	[6] hunterFormsBS.bandSplitRotator.BandSplitRotator

	[7] hunterFormsBS.bs_roformer.BSRoformer

	[8] hunterFormsBS.mel_band_roformer.MelBandRoformer

	[9] PyTorch.
		https://context7.com/pytorch/pytorch
	"""
	def __init__(
		self,
		dim: int,
		dim_head: int = 64,
		attn_dropout: float = 0.0,
		heads: int = 8,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
		scale: float | None = None,
		*,
		flash: bool = True,
		sage_attention: bool = False,
		use_value_residual_learning: bool = False,
		learned_value_residual_mix: bool | None = None,
	) -> None:
		"""Set up an attention block for a chosen width and head layout.

		You can use `__init__` to choose the feature width, head layout, dropout probability,
		position-encoding path, attention backend, and optional value-residual-learning path for the
		block. `__init__` stores the resulting submodules so later calls to `forward` can reuse the
		same attention block. `__init__` also preserves the compatibility switch
		`learned_value_residual_mix` so wrapper classes can forward older configuration names without
		translation [1][2][3].

		PyTorch
		-------
		submodule construction : implementation detail
			`__init__` computes `dim_inner = heads * dim_head`, stores `rotary_embed` and
			`pope_embed`, creates `Attend(flash=flash, attn_dropout=attn_dropout,
			sage_attention=sage_attention)` [4], instantiates `RMSNorm(dim)`, and constructs `to_qkv =
			nn.Linear(dim, dim_inner * 3, bias=False)`, `to_gates = nn.Linear(dim, heads)`, and
			`to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(attn_dropout))`
			[5]. When `use_value_residual_learning` is `True` and `learned_value_residual_mix` is
			`None`, or when `learned_value_residual_mix` is `True`, `__init__` also creates
			`self.learned_value_residual_mix = nn.Linear(dim, heads)` [5].

		Parameters
		----------
		dim : int
			Input feature width before projection and output feature width after `to_out`.
		heads : int = 8
			Number of attention heads.
		dim_head : int = 64
			Feature width of each head before concatenation.
		attn_dropout : float = 0.0
			Probability used by `Attend` [4] and the output dropout layer.
		pope_embed : PoPE | None = None
			Optional polar-coordinate position encoder [6][7]. When `pope_embed` is not `None`,
			`forward` prefers the PoPE path over the rotary path.
		rotary_embed : RotaryEmbedding | None = None
			Optional rotary position encoder [8]. `forward` uses `rotary_embed` only when `pope_embed`
			is `None`.
		scale : float | None = None
			Optional attention-score multiplier override. When `scale` is `None`, `__init__` stores
			`dim_head ** -0.5`.
		flash : bool = True
			Whether `Attend` [4] may use PyTorch SDPA [5] when the PoPE path is not active and
			`sage_attention` is `False`.
		sage_attention : bool = False
			Whether `Attend` [4] should prefer SageAttention [9][10]. `hunterFormsBS` does not install
			`sageattention`.
		use_value_residual_learning : bool = False
			Whether `__init__` should create `self.learned_value_residual_mix` for `forward`.
		learned_value_residual_mix : bool | None = None
			Compatibility switch kept for older configuration files. `True` also creates
			`self.learned_value_residual_mix`.

		References
		----------
		[1] hunterFormsBS.bandSplitRotator.BandSplitRotator

		[2] hunterFormsBS.bs_roformer.BSRoformer

		[3] hunterFormsBS.mel_band_roformer.MelBandRoformer

		[4] hunterFormsBS.attend.Attend

		[5] PyTorch.
			https://context7.com/pytorch/pytorch
		[6] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
			Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings.
			https://arxiv.org/abs/2509.10534
		[7] lucidrains/PoPE-pytorch.
			https://github.com/lucidrains/PoPE-pytorch
		[8] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
			RoFormer: Enhanced Transformer with Rotary Position Embedding.
			https://arxiv.org/abs/2104.09864
		[9] Zhang, J., Wei, J., Huang, H., Zhang, P., Zhu, J., and Chen, J. (2025).
			SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.
			https://arxiv.org/abs/2410.02367
		[10] thu-ml/SageAttention
			https://github.com/thu-ml/SageAttention
		"""
		super().__init__()

		self.heads: int = heads
		self.norm: RMSNorm = RMSNorm(dim)
		self.pope_embed: PoPE | None = pope_embed
		self.rotary_embed: RotaryEmbedding | None = rotary_embed
		self.scale: float = scale or dim_head ** neg(0.5)

		self.attend: Attend = Attend(attn_dropout=attn_dropout, scale=self.scale, flash=flash, sage_attention=sage_attention)
		self.to_gates: nn.Linear = nn.Linear(dim, self.heads)

		dim_inner: int = self.heads * dim_head

		self.to_qkv: nn.Linear = nn.Linear(in_features=dim, out_features=dim_inner * 3, bias=False)
		self.to_out: nn.Sequential = nn.Sequential(nn.Linear(in_features=dim_inner, out_features=dim, bias=False), nn.Dropout(attn_dropout))

		self.learned_value_residual_mix: nn.Linear | None = None

		if ((use_value_residual_learning is True and learned_value_residual_mix is None)
			or (learned_value_residual_mix is True)):
			self.learned_value_residual_mix = nn.Linear(dim, self.heads)

		"""Original code because I have not tested the new code:
		if exists(learned_value_residual_mix):
			use_value_residual_learning = learned_value_residual_mix

		self.learned_value_residual_mix = nn.Linear(dim, self.heads) if use_value_residual_learning else None"""

	@overload
	def forward(self, x: Tensor, value_residual: None = None) -> Tensor:...
	@overload
	def forward(self, x: Tensor, value_residual: Tensor) -> tuple[Tensor, Tensor | None]:...
	def forward(self, x: Tensor, value_residual: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor | None]:
		"""Compute gated multi-head attention output from activations `x`.

		You can use `forward` to normalize activations `x`, project activation `Tensor` `x` into query
		`Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v`, optionally mix `value_residual` into
		value `Tensor` `v` with a learned per-head gate, inject position through `pope_embed` [1][2]
		or `rotary_embed` [3], aggregate value `Tensor` `v` through `Attend.forward` [4] or the
		PoPE-specific flash-attention path [2], and return the gated, concatenated, projected result.
		When `value_residual` is not `None`, `forward` also returns the unmixed value `Tensor`
		captured before the residual mix so outer stacks such as `Transformer` [5] and
		`BandSplitRotator` [6] can reuse that value in later blocks.

		Parameters
		----------
		x : Tensor
			Input activation `Tensor` with shape `batch × sequence position × feature`.
		value_residual : Tensor | None = None
			Optional value `Tensor` from an earlier block. `forward` uses `value_residual` only when
			`self.learned_value_residual_mix` is not `None`.

		Returns
		-------
		out : Tensor | tuple[Tensor, Tensor | None]
			When `value_residual` is `None`, `out` is the output activation `Tensor` with shape `batch
			× sequence position × feature`. When `value_residual` is not `None`, `out` is `(out,
			original_values)`, where `original_values` is the pre-mixed value `Tensor`.

		Mathematics
		-----------
		gated multi-head attention : equation
		```
			Let H ≜ `self.heads`,  α ≜ `self.scale`,
				X ≜ `self.norm(x)`,  Q ≜ `q`,  K ≜ `k`,  V₀ ≜ `original_values`,
				R ≜ `value_residual`,  Λ ≜ `self.learned_value_residual_mix`,
				M ≜ σ(rearrange(Λ(X), 'b n h -> b h n 1')),
				wₕᴳ ≜ `to_gates`.weight[h, :],  h ∈ {0, …, H−1},
				Wᴼ ≜ `to_out[0]`.weight

			Λ = ∅ ∨ R = ∅  ⟹  V = V₀
			Λ ≠ ∅ ∧ R ≠ ∅  ⟹  V = (1 − M) ⊙ V₀ + M ⊙ R

			S[b, h, i, j] = ⟨Q[b, h, i, :], K[b, h, j, :]⟩ · α
			A[b, h, i, j] = exp(S[b, h, i, j]) / ∑ₘ exp(S[b, h, i, m])
			O[b, h, i, :] = ∑ⱼ A[b, h, i, j] · V[b, h, j, :]
			G[b, i, h] = σ(⟨X[b, i, :], wₕᴳ⟩)
			Z[b, i, hd : (h+1)d] = G[b, i, h] · O[b, h, i, :]
			Y[b, i, :] = Z[b, i, :] Wᴼ

			where  G ≜ `gates`,  Y ≜ `out`
		```

		PyTorch
		-------
		path selection and head gating : implementation detail
			`forward` computes query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v` from
			normalized `x`. When `pope_embed` is available, `forward` calls `flash_attn_with_pope` [2]
			with `pos_emb=self.pope_embed(x.shape[-2])` and `softmax_scale=self.scale`. When
			`pope_embed` is absent and `rotary_embed` is available, `forward` rotates query `Tensor`
			`q` and key `Tensor` `k` before `Attend.forward` [4]. When neither positional encoder is
			available, `forward` passes query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v`
			directly to `Attend.forward` [4]. After the attention output is produced, `forward`
			multiplies each head output by `sigmoid(self.to_gates(X))`, concatenates the gated heads,
			and applies `to_out`.

		References
		----------
		[1] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
			Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings.
			https://arxiv.org/abs/2509.10534
		[2] lucidrains/PoPE-pytorch.
			https://github.com/lucidrains/PoPE-pytorch
		[3] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
			RoFormer: Enhanced Transformer with Rotary Position Embedding.
			https://arxiv.org/abs/2104.09864
		[4] hunterFormsBS.attend.Attend.forward

		[5] hunterFormsBS.attend.Transformer

		[6] hunterFormsBS.bandSplitRotator.BandSplitRotator

		[7] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
			Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need.
			https://arxiv.org/abs/1706.03762
		"""
		x = self.norm(x)

		q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

		original_values: Tensor = v

		if exists(self.learned_value_residual_mix):
			mix: Tensor = self.learned_value_residual_mix(x)
			mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()
			v: Tensor = v.lerp(raiseIfNone(value_residual), mix)

		if exists(self.pope_embed):
			out = flash_attn_with_pope(q, k, v, pos_emb=self.pope_embed(q.shape[-2]), softmax_scale=self.scale)
		elif exists(self.rotary_embed):
			q: Tensor = self.rotary_embed.rotate_queries_or_keys(q)
			k: Tensor = self.rotary_embed.rotate_queries_or_keys(k)
			out = self.attend(q, k, v)
		else:
			out = self.attend(q, k, v)

		# after attend
		gates: Tensor = self.to_gates(x)
		out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

		out = rearrange(out, 'b h n d -> b n (h d)')
		out = self.to_out(out)
		if exists(value_residual):
			out = (out, original_values)
		return out

class FeedForward(Module):
	"""Transform activations with a position-wise expansion-and-projection block.

	You can use `FeedForward` after `Attention` [1] inside `Transformer` [2] when activation `Tensor`
	`x` at each sequence position should pass through the same nonlinear feature transform.
	`FeedForward` preserves the batch axis, sequence axis, and feature axis of activation `Tensor` `x`
	while expanding the feature width internally before projecting back to the original width. The
	shared `BandSplitRotator`, `BSRoformer`, and `MelBandRoformer` stacks use one `FeedForward` block
	after every attention block [3][4][5].

	Attributes
	----------
	net : nn.Sequential
		Position-wise submodule sequence containing root-mean-square normalization, width expansion,
		`nn.GELU`, `ff_dropout`, width projection, and output `ff_dropout` [6].

	See Also
	--------
	Attention
		Provide the sequence-mixing sublayer that precedes `FeedForward` inside `Transformer`.
	Transformer
		Stack `Attention` with `FeedForward` and residual logic.

	References
	----------
	[1] hunterFormsBS.attend.Attention

	[2] hunterFormsBS.attend.Transformer

	[3] hunterFormsBS.bandSplitRotator.BandSplitRotator

	[4] hunterFormsBS.bs_roformer.BSRoformer

	[5] hunterFormsBS.mel_band_roformer.MelBandRoformer

	[6] PyTorch.
		https://context7.com/pytorch/pytorch
	"""
	def __init__(self, dim: int, ff_mult: float | None = 4.0, ff_dropout: float = 0.0) -> None:
		"""Set up a position-wise feedforward block for feature width `dim`.

		You can use `__init__` to choose the input and output feature width `dim`, the internal width
		expansion factor `ff_mult`, and the `ff_dropout` probability for the shared feedforward
		sublayer inside `Transformer` [1]. `__init__` stores the resulting normalization, linear,
		activation, and `ff_dropout` layers in `net` for reuse by later calls to `forward`.

		Parameters
		----------
		dim : int
			Input feature width before expansion and output feature width after projection.
		ff_mult : float | None = 4.0
			Multiplicative factor used to compute the hidden feature width `int(dim * ff_mult)`. When
			`ff_mult` is `None`, `__init__` uses `4.0`.
		ff_dropout : float = 0.0
			Dropout probability applied after the hidden activation and after the output projection.

		PyTorch
		-------
		submodule construction : implementation detail
			`__init__` computes `dim_inner = int(dim * ff_mult)` and stores `net` as a six-stage
			`nn.Sequential` with root-mean-square normalization, `nn.Linear(dim, dim_inner)`,
			`nn.GELU`, `nn.Dropout(ff_dropout)`, `nn.Linear(dim_inner, dim)`, and a second
			`nn.Dropout(ff_dropout)` [2].

		References
		----------
		[1] hunterFormsBS.attend.Transformer

		[2] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		super().__init__()
		if ff_mult is None:
			ff_mult = 4.0
		dim_inner: int = int(dim * ff_mult)
		self.net: nn.Sequential = nn.Sequential(
			RMSNorm(dim), nn.Linear(dim, dim_inner), nn.GELU(), nn.Dropout(ff_dropout), nn.Linear(dim_inner, dim), nn.Dropout(ff_dropout)
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Transform activations `x` with the position-wise feedforward block.

		You can use `forward` when `x` already has batch axis, sequence axis, and feature axis.
		`forward` applies the same nonlinear feature transform to every position independently, reuses
		the submodule sequence configured by `__init__`, and returns an activation `Tensor` with the
		same outer shape as `x`. `forward` supplies the position-wise branch used by `Transformer`
		[1].

		Parameters
		----------
		x : Tensor
			Input activation `Tensor` with shape `batch × sequence position × feature`.

		Returns
		-------
		out : Tensor
			Output activation `Tensor` with shape `batch × sequence position × feature`.

		Shape Transformation
		--------------------
		position-wise mapping : implementation detail
			`forward` preserves the outer shape `B×N×D`. `forward` applies the same feedforward map to
			each position vector `x[b, n, :]` independently of every other position.

		References
		----------
		[1] hunterFormsBS.attend.Transformer
		"""
		return self.net(x)

class Transformer(Module):
	"""Refine activations with repeated attention-and-feedforward blocks.

	You can use `Transformer` as the shared sequence stack inside `BandSplitRotator` [1], `BSRoformer`
	[2], and `MelBandRoformer` [3]. `Transformer` takes activation `Tensor` `x`, repeats `depth` pairs
	of `Attention` [4] and `FeedForward` [5], optionally installs multi-stream residual adapters, and
	optionally normalizes the result. `Transformer` preserves the batch axis, sequence axis, and
	feature axis of activation `Tensor` `x`. The constructor still accepts several compatibility
	parameters from older wrapper configurations, but the stack itself no longer contains
	`LinearAttention`.

	Attributes
	----------
	layers : ModuleList
		Sequence of per-depth `[attention, feedforward]` pairs, where the attention entry is
		`Attention` [4] and the feedforward entry is `FeedForward` [5].
	norm : RMSNorm | nn.Identity
		Final output normalization module applied after the last residual block when `norm_output` is
		`True`, or `nn.Identity` otherwise.

	See Also
	--------
	Attention
		Provide the default sequence-position attention branch used by `Transformer`.
	FeedForward
		Provide the position-wise feature transform paired with each attention layer.

	References
	----------
	[1] hunterFormsBS.bandSplitRotator.BandSplitRotator

	[2] hunterFormsBS.bs_roformer.BSRoformer

	[3] hunterFormsBS.mel_band_roformer.MelBandRoformer

	[4] hunterFormsBS.attend.Attention

	[5] hunterFormsBS.attend.FeedForward
	"""
	def __init__(
		self,
		*,
		depth: int,
		attn_dropout: float = 0.0,
		dim_head: int = 64,
		dim: int,
		ff_dropout: float = 0.0,
		ff_mult: float | None = 4,
		flash_attn: bool = True,
		heads: int = 8,
		learned_value_residual_mix: bool | None = None,
		linear_attn: bool = False,  # noqa: ARG002
		mc_hyper_conn_sinkhorn_iters: int | None = None,  # noqa: ARG002
		norm_output: bool = True,
		num_residual_fracs: int | None = None,  # noqa: ARG002
		num_residual_streams: int = 1,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
		sage_attention: bool = False,
		scale: float | None = None,
		use_value_residual_learning: bool = False,
	) -> None:
		"""Set up a transformer stack for feature width `dim` and layer count `depth`.

		You can use `__init__` to choose the stack depth, model width, attention-head layout, dropout
		probabilities, attention backend, position-encoding path, and residual-stream behavior for a
		shared transformer block. `__init__` stores `depth` pairs of `Attention` [1] and `FeedForward`
		[2] in `layers`, forwards the downstream attention parameters without renaming through
		`ParametersAttention` [3], and configures optional output normalization in `norm`. `__init__`
		retains `linear_attn`, `mc_hyper_conn_sinkhorn_iters`, and `num_residual_fracs` only for
		configuration compatibility with earlier wrappers.

		Parameters
		----------
		attn_dropout : float = 0.0
			Dropout probability passed to each attention sublayer.
		depth : int
			Number of repeated attention-plus-feedforward residual blocks.
		dim_head : int = 64
			Feature width of each attention head.
		dim : int
			Input and output feature width of the stack.
		ff_dropout : float = 0.0
			Dropout probability passed to each `FeedForward` block [2].
		ff_mult : float | None = 4
			Hidden-width expansion factor passed to each `FeedForward` block [2].
		flash_attn : bool = True
			Whether each attention sublayer may use PyTorch SDPA (scaled dot-product attention)
			backends [4].
		heads : int = 8
			Number of attention heads in each attention sublayer.
		learned_value_residual_mix : bool | None = None
			Compatibility switch passed to each `Attention` [1]. `True` asks each `Attention` to
			create the learned value-residual mixing projection.
		linear_attn : bool = False
			Vestigial flag retained for wrapper configuration passthrough. This unified implementation
			ignores `linear_attn` because `LinearAttention` was removed.
		mc_hyper_conn_sinkhorn_iters : int | None = None
			Compatibility parameter retained for wrapper configuration passthrough. `__init__` ignores
			`mc_hyper_conn_sinkhorn_iters`.
		norm_output : bool = True
			Whether to apply final output normalization after the last residual block.
		num_residual_fracs : int | None = None
			Compatibility parameter retained for wrapper configuration passthrough. `__init__` ignores
			`num_residual_fracs`.
		num_residual_streams : int = 1
			Number of residual streams to prepare. When `num_residual_streams != 1`, `__init__` wraps
			each stored sublayer with a residual-stream adapter after construction.
		pope_embed : PoPE | None = None
			Optional polar-coordinate position encoder passed to each `Attention` [1] [5][6]. When
			both positional encoders are present, `Attention.forward` prefers `pope_embed`.
		rotary_embed : RotaryEmbedding | None = None
			Optional rotary positional encoder passed to each `Attention` [1] [7].
		sage_attention : bool = False
			Whether each `Attention` should prefer SageAttention [8][9]. `hunterFormsBS` does not
			install `sageattention`.
		scale : float | None = None
			Optional attention-score multiplier override passed to each `Attention` [1].
		use_value_residual_learning : bool = False
			Whether each `Attention` should create and use the learned value-residual mixing path.

		PyTorch
		-------
		layer construction : implementation detail
			`__init__` first builds `ParametersAttention` [3] with the downstream attention arguments.
			`__init__` then creates `depth` `ModuleList` pairs, each containing one
			`Attention(**parametersAttention)` [1] and one `FeedForward(dim=dim, ff_mult=ff_mult,
			ff_dropout=ff_dropout)` [2]. When `num_residual_streams != 1`, `__init__` wraps each
			stored sublayer with a residual-stream adapter. `norm` becomes `RMSNorm(dim)` when
			`norm_output` is `True` and `nn.Identity()` otherwise [4].

		References
		----------
		[1] hunterFormsBS.attend.Attention

		[2] hunterFormsBS.attend.FeedForward

		[3] hunterFormsBS.theTypes.ParametersAttention

		[4] PyTorch.
			https://context7.com/pytorch/pytorch
		[5] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
			Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings.
			https://arxiv.org/abs/2509.10534
		[6] lucidrains/PoPE-pytorch.
			https://github.com/lucidrains/PoPE-pytorch
		[7] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
			RoFormer: Enhanced Transformer with Rotary Position Embedding.
			https://arxiv.org/abs/2104.09864
		[8] Zhang, J., Wei, J., Huang, H., Zhang, P., Zhu, J., and Chen, J. (2025).
			SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.
			https://arxiv.org/abs/2410.02367
		[9] thu-ml/SageAttention
			https://github.com/thu-ml/SageAttention
		"""
		super().__init__()

		parametersAttention = ParametersAttention(
			attn_dropout=attn_dropout,
			dim_head=dim_head,
			dim=dim,
			flash=flash_attn,
			heads=heads,
			learned_value_residual_mix=learned_value_residual_mix,
			pope_embed=pope_embed,
			rotary_embed=rotary_embed,
			sage_attention=sage_attention,
			scale=scale,
			use_value_residual_learning=use_value_residual_learning,
		)

		self.layers = ModuleList(
			ModuleList([Attention(**parametersAttention)
					, FeedForward(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout)])
				for _deep in loops(depth)
		)

		if num_residual_streams != 1:
			init_hyper_conn, *_streams = get_init_and_expand_reduce_stream_functions(
				num_residual_streams, disable=(num_residual_streams == 1)
			)
			for moduleList in self.layers:
				moduleList = cast('ModuleList', moduleList)
				for nnModule in moduleList:
					init_hyper_conn(dim=dim, branch=nnModule)

		"""Original code because I have not tested the new code:
		self.layers = ModuleList([])
		for _deep in loops(depth):
			attn = Attention(**parametersAttention, use_value_residual_learning=use_value_residual_learning)
			if num_residual_streams != 1:
				attn = init_hyper_conn(dim=dim, branch=attn)

			ff = FeedForward(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout)
			if num_residual_streams != 1:
				ff = init_hyper_conn(dim=dim, branch=ff)

			self.layers.append(ModuleList([attn, ff]))"""

		self.norm: RMSNorm | nn.Identity = RMSNorm(dim) if norm_output else nn.Identity()

	@overload
	def forward(self, x: Tensor, value_residual: None = None) -> Tensor:...
	@overload
	def forward(self, x: Tensor, value_residual: Tensor) -> tuple[Tensor, Tensor | None]:...
	def forward(self, x: Tensor, value_residual: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor | None]:
		"""Transform activations `x` through the residual stack.

		You can use `forward` when `x` already has batch axis, sequence axis, and feature axis. When
		`value_residual` is `None`, `forward` applies each stored `Attention` [1] and `FeedForward`
		[2] with residual additions and then applies `norm`. When `value_residual` is not `None`,
		`forward` passes `value_residual` to each stored `Attention`, keeps the first returned
		pre-mixed value `Tensor`, applies each stored `FeedForward`, and returns the normalized
		activation together with that first value `Tensor`. The pair-returning path is the
		value-residual-learning path consumed by the wrapper model classes [3][4][5].

		Parameters
		----------
		x : Tensor
			Input activation `Tensor` with shape `batch × sequence position × feature`.
		value_residual : Tensor | None = None
			Optional value `Tensor` passed to each stored `Attention`.

		Returns
		-------
		result : Tensor | tuple[Tensor, Tensor | None]
			When `value_residual` is `None`, `result` is the output activation `Tensor` with shape
			`batch × sequence position × feature`. When `value_residual` is not `None`, `result` is
			`(transformed_x, first_values)`, where `first_values` is the pre-mixed value `Tensor` from
			the first attention block, or `None` when `self.layers` is empty.

		Mathematics
		-----------
		residual recurrence : equation
		```
			Let L ≜ `len(self.layers)`,  X₀ ≜ `x`,
				Aₗ ≜ `self.layers[ℓ][0]`,  Fₗ ≜ `self.layers[ℓ][1]`,  ℓ ∈ {0, …, L−1},
				N ≜ `self.norm`

			Hₗ = Aₗ(Xₗ) + Xₗ
			Xₗ₊₁ = Fₗ(Hₗ) + Hₗ
			Y = N(X_L)

			where  Y ≜ `x`
		```

		value residual learning : equation
		```
			Let L ≜ `len(self.layers)`,  X₀ ≜ `x`,  R ≜ `value_residual`,
				Aₗ ≜ `self.layers[ℓ][0]`,  Fₗ ≜ `self.layers[ℓ][1]`,  ℓ ∈ {0, …, L−1},
				N ≜ `self.norm`

			(Ĥₗ, Vₗ) = Aₗ(Xₗ; R)
			Xₗ₊₁ = Fₗ(Ĥₗ)
			Y = N(X_L)
			U = V₀

			where  Y ≜ `x`,  U ≜ `first_values`
		```

		References
		----------
		[1] hunterFormsBS.attend.Attention.forward

		[2] hunterFormsBS.attend.FeedForward.forward

		[3] hunterFormsBS.bandSplitRotator.BandSplitRotator

		[4] hunterFormsBS.bs_roformer.BSRoformer

		[5] hunterFormsBS.mel_band_roformer.MelBandRoformer
		"""
		first_values: Tensor | None = None
		if value_residual is not None:
			for sherpa in self.layers:
				attn: Attention = cast('Attention', cast('ModuleList', sherpa)[0])
				ff: FeedForward = cast('FeedForward', cast('ModuleList', sherpa)[1])
				x, next_values = attn(x, value_residual=value_residual)
				first_values = default(first_values, next_values)
				x = ff(x)
		else:
			for sherpa in self.layers:
				attn: Attention = cast('Attention', cast('ModuleList', sherpa)[0])
				ff: FeedForward = cast('FeedForward', cast('ModuleList', sherpa)[1])
				x = attn(x) + x
				x = ff(x) + x

		x = self.norm(x)

		if value_residual is not None:
			return x, first_values
		return x
