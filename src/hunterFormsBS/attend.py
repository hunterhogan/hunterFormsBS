# ruff: noqa: PLC0415
"""Provide attention, feedforward, and transformer blocks for music source separation.

You can use this module to assemble the hierarchical attention stack shared by BS-RoFormer [1] and
Mel-Band RoFormer [2]. The module provides four building-block classes. `Attend` evaluates attention
from precomputed query, key, and value arrays and optionally delegates to PyTorch SDPA backends [3].
`Attention` projects activations into query, key, and value arrays, applies RoPE [4] or PoPE [5]
positional encoding, gates head outputs, and calls `Attend`. `FeedForward` applies a position-wise
expansion-and-projection block after each attention sublayer. `LinearAttention` provides the optional
XCiT-style cross-covariance attention pre-block [6]. `Transformer` stacks those blocks into the
repeated attention-and-feedforward sequence consumed by `BandSplitRotator` [7].

Contents
--------
Classes
	Attend
		Evaluate attention output from precomputed query, key, and value arrays using an explicit or
		PyTorch SDPA path.
	Attention
		Project activations into multi-head query, key, and value arrays, apply positional encoding,
		and return gated attention output.
	FeedForward
		Apply a position-wise expansion-and-projection nonlinear block.
	LinearAttention
		Apply XCiT-style cross-covariance attention across feature channels.
	Transformer
		Stack attention and feedforward sublayers into a repeated residual sequence.

References
----------
[1] Lu, W.-T., Wang, J.-C., Kong, Q., & Hung, Y.-N. (2023). Music Source Separation with
	Band-Split RoPE Transformer. https://arxiv.org/abs/2309.02612
[2] Wang, J.-C., Lu, W.-T., & Won, M. (2023). Mel-Band RoFormer for Music Source Separation.
	https://arxiv.org/abs/2409.04702
[3] PyTorch.
	https://context7.com/pytorch/pytorch
[4] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced
	Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864
[5] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., & Mozer, M. C. (2025). Decoupling the
	"What" and "Where" With Polar Coordinate Positional Embeddings. https://arxiv.org/abs/2509.10534
[6] El-Nouby, A., Touvron, H., Caron, M., Bojanowski, P., Douze, M., Joulin, A., Laptev, I.,
	Neverova, N., Synnaeve, G., Verbeek, J., & Jégou, H. (2021). XCiT: Cross-Covariance Image
	Transformers. https://arxiv.org/abs/2106.09681
[7] hunterFormsBS.bandSplitRotator.BandSplitRotator
"""
from __future__ import annotations

from einops import rearrange
from hunterFormsBS.theTypes import FlashAttentionConfig, KwargsOfAttention
from more_itertools import loops
from operator import neg
from packaging import version
from PoPE_pytorch import flash_attn_with_pope, PoPE
from torch import einsum, nn, Tensor
from torch.nn import Module, ModuleList
from torch_einops_kit import exists, once
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
	from rotary_embedding_torch import RotaryEmbedding
	from torch._C import _CudaDeviceProperties

print_once = once(print)

class Attend(nn.Module):
	"""Compute attention output from precomputed query, key, and value arrays.

	You can use `Attend` after a higher-level block such as `Attention` [1] or
	`LinearAttention` [2] has already built query `Tensor` `q`, key `Tensor` `k`, and value
	`Tensor` `v`. `Attend` compares query `Tensor` `q` against key `Tensor` `k` to build attention
	weights, then uses those attention weights to mix value `Tensor` `v`. `Attend` can compute
	that mapping with an explicit scaled dot-product attention implementation [4] or with PyTorch
	SDPA (scaled dot-product attention) backends [5][6].

	Attributes
	----------
	attn_dropout : nn.Dropout
		Dropout module applied to attention weights after `softmax` in the explicit path [5].
	dropout : float
		Probability used for attention-weight dropout during training.
	cpu_config : FlashAttentionConfig
		Backend flags passed to `torch.backends.cuda.sdp_kernel` [5] when execution remains on CPU or
		when `cuda_config` is unavailable.
	cuda_config : FlashAttentionConfig | None = None
		Backend flags used for CUDA execution when `flash` is enabled. `cuda_config` remains `None`
		until `Attend` discovers a usable CUDA device during `__init__`.
	flash : bool
		Whether `forward` should delegate to `flash_attn`.
	scale : float = q.shape[-1] ** -0.5
		Attention-score multiplier. By default, `Attend` uses `q.shape[-1] ** -0.5`, the standard
		inverse-square-root factor for the query feature width [4].

	See Also
	--------
	Attention
		Project model activations into query, key, and value arrays for standard multi-head
		attention.
	LinearAttention
		Project model activations into normalized query, key, and value arrays for the
		linear-attention variant.

	References
	----------
	[1] hunterFormsBS.attend.Attention

	[2] hunterFormsBS.attend.LinearAttention

	[3] hunterFormsBS.theTypes.FlashAttentionConfig

	[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
		Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need.
		https://arxiv.org/abs/1706.03762
	[5] PyTorch.
		https://context7.com/pytorch/pytorch
	[6] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022).
		FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
		https://arxiv.org/abs/2205.14135
	"""

	def __init__(self, dropout: float, scale: float, *, flash: bool = False, sage_attention: bool = False) -> None:
		"""Configure attention-score scaling, dropout, and scaled dot-product attention backend selection.

		You can use `__init__` to set the dropout probability for attention weights, optionally
		override the default inverse-square-root score scaling `q.shape[-1] ** -0.5` [1], and allow
		`forward` to use PyTorch SDPA (scaled dot-product attention) backends [2]. `__init__`
		stores one backend configuration for CPU execution and, when `flash=True` and a supported
		device is available, one backend configuration for CUDA execution. `__init__` raises
		`RuntimeError` when `flash=True` with PyTorch earlier than 2.0.0 [2].

		Parameters
		----------
		dropout : float = 0.0
			Probability applied to attention weights during training.
		scale : float | None = q.shape[-1] ** -0.5
			Attention-score multiplier override. Passing `None` keeps the default
			`q.shape[-1] ** -0.5` [1].
		flash : bool = False
			Whether `forward` may delegate to PyTorch SDPA (scaled dot-product attention) backends
			[2][3].

		PyTorch
		-------
		backend selection : implementation detail
			`__init__` inspects the active CUDA device when `flash` is `True`. `__init__` enables the
			flash backend only on A100 devices and keeps the math plus memory-efficient backends
			available on other CUDA devices [2][3].

		References
		----------
		[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
			Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need.
			https://arxiv.org/abs/1706.03762
		[2] PyTorch.
			https://context7.com/pytorch/pytorch
		[3] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022).
			FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
			https://arxiv.org/abs/2205.14135
		"""
		super().__init__()
		self.scale: float = scale
		self.dropout: float = dropout
		self.attn_dropout: nn.Dropout = nn.Dropout(self.dropout)

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
		"""Compute attention output from query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v` with PyTorch scaled dot-product attention.

		You can use `flash_attn` when query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v`
		already share compatible batch, head, sequence, and head-feature axes. `flash_attn`
		rescales query `Tensor` `q` when `self.scale` overrides the default inverse-square-root
		query-feature normalization [2]. `flash_attn` then selects a PyTorch SDPA
		(scaled dot-product attention) backend with `torch.backends.cuda.sdp_kernel` [1] and returns
		the attention-weighted mixture of value `Tensor` `v` through
		`torch.nn.functional.scaled_dot_product_attention` [1]. When the selected backend supports
		it, PyTorch may realize the computation with a flash-style kernel [3].

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

			where  A ≜ implicit attention matrix,  O ≜ `out`,  S ≜ implicit score matrix
		```

		PyTorch
		-------
		backend selection : implementation detail
			`flash_attn` always enters `torch.backends.cuda.sdp_kernel` [1] before calling
			`torch.nn.functional.scaled_dot_product_attention` [1]. `cpu_config` enables all three
			SDPA backend families. When `q.is_cuda` is `True` and `cuda_config` is available,
			`flash_attn` switches to the stored CUDA-specific flags. During training, `flash_attn`
			passes `self.dropout`, and during evaluation, `flash_attn` passes `0.0`.

		References
		----------
		[1] PyTorch.
			https://context7.com/pytorch/pytorch
		[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
			Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need.
			https://arxiv.org/abs/1706.03762
		[3] Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. (2022).
			FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
			https://arxiv.org/abs/2205.14135
		"""
		is_cuda: bool = q.is_cuda

		q = q * (self.scale / (q.shape[-1] ** neg(0.5)))

		# Check if there is a compatible device for flash attention
		config: FlashAttentionConfig = self.cpu_config
		if is_cuda and exists(self.cuda_config):
			config = self.cuda_config

		# pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale  # noqa: ERA001
		with torch.backends.cuda.sdp_kernel(**config._asdict()): # pyright: ignore[reportDeprecated]
			out: Tensor = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)

		return out

	def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
		"""Compute attention output from query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v`.

		You can use `forward` after another module has already built query `Tensor` `q`, key
		`Tensor` `k`, and value `Tensor` `v`. `forward` compares query `Tensor` `q` against key
		`Tensor` `k` to build attention weights, then uses those attention weights to mix value
		`Tensor` `v`. When `self.flash` is `True`, `forward` delegates to `flash_attn` [1].
		Otherwise `forward` computes the standard scaled dot-product attention weights [2]
		explicitly with `softmax`, applies `attn_dropout`, and mixes value `Tensor` `v` with those
		attention weights.

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

			S[b, h, i, j] = ⟨Q[b, h, i, :], K[b, h, j, :]⟩ · s
			A[b, h, i, j] = exp(S[b, h, i, j]) / ∑ₘ exp(S[b, h, i, m])
			O[b, h, i, :] = ∑ⱼ A[b, h, i, j] · V[b, h, j, :]

			where  S ≜ `similarity`,  A ≜ `attention_weights`,  O ≜ returned value
		```

		References
		----------
		[1] hunterFormsBS.attend.Attend.flash_attn

		[2] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
			Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need.
			https://arxiv.org/abs/1706.03762
		"""
		if self.sage_attention:
			"""At the moment, you need to install `sageattention` manually.

			https://github.com/thu-ml/SageAttention

			@inproceedings{zhang2025sageattention,
			title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration},
			author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
			booktitle={International Conference on Learning Representations (ICLR)},
			year={2025}
			}
			@inproceedings{zhang2024sageattention2,
			title={Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization},
			author={Zhang, Jintao and Huang, Haofeng and Zhang, Pengle and Wei, Jia and Zhu, Jun and Chen, Jianfei},
			booktitle={International Conference on Machine Learning (ICML)},
			year={2025}
			}
			@article{zhang2025sageattention2++,
			title={Sageattention2++: A more efficient implementation of sageattention2},
			author={Zhang, Jintao and Xu, Xiaoming and Wei, Jia and Huang, Haofeng and Zhang, Pengle and Xiang, Chendong and Zhu, Jun and Chen, Jianfei},
			journal={arXiv preprint arXiv:2505.21136},
			year={2025}
			}
			@article{zhang2025sageattention3,
			title={SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training},
			author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Xu, Xiaoming and Huang, Haofeng and Wang, Haoxu and Jiang, Kai and Zhu, Jun and Chen, Jianfei},
			journal={arXiv preprint arXiv:2505.11594},
			year={2025}
			}
			"""
			from sageattention import sageattn  # pyright: ignore[reportMissingImports, reportUnknownVariableType] # ty:ignore[unresolved-import]
			return sageattn(q, k, v, tensor_layout='HND', is_causal=False) # pyright: ignore[reportUnknownVariableType]
		if self.flash:
			return self.flash_attn(q, k, v)

		similarity: Tensor = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

		attention_weights: Tensor = similarity.softmax(dim=-1)
		attention_weights = self.attn_dropout(attention_weights)

		# aggregate values

		return einsum('b h i j, b h j d -> b h i d', attention_weights, v)

class Attention(nn.Module):
	"""Mix information across sequence positions and return updated features.

	You can use `Attention` when activation `Tensor` `x` at each sequence position should gather
	information from other sequence positions before the next block. `Attention` projects
	activation `Tensor` `x` into query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v`,
	optionally injects position through `rotary_embed` [2] or `pope_embed` [3][4], and returns one
	updated feature vector for each input position. The output keeps the same outer shape as
	activation `Tensor` `x`, so `Attention` fits naturally inside `Transformer` [5] and the
	related source-separation models [6][7].

	PyTorch
	-------
	module structure : implementation detail
		`Attention` applies `RMSNorm` to the input activations, projects the normalized activations
		with `to_qkv`, reshapes the projected result into head-specific query `Tensor` `q`, key
		`Tensor` `k`, and value `Tensor` `v`, and then chooses one of three execution paths.
		`Attention` calls `flash_attn_with_pope` [4] when `pope_embed` is available, rotates query
		`Tensor` `q` and key `Tensor` `k` before `Attend` [1] when `rotary_embed` is available, and
		otherwise delegates directly to `Attend` [1]. After aggregation, `Attention` multiplies each
		head output by a sigmoid gate from `to_gates`, concatenates the gated head outputs, and
		applies `to_out`.

	Attributes
	----------
	attend : Attend
		Exact attention core used when `pope_embed` is `None` [1].
	heads : int
		Number of attention heads.
	norm : RMSNorm
		Root-mean-square normalization module applied before the projections.
	pope_embed : PoPE | None
		Optional polar coordinate position encoder [3][4].
	rotary_embed : RotaryEmbedding | None
		Optional rotary position encoder [2].
	scale : float = dim_head ** -0.5
		Head-feature normalization factor stored as `dim_head ** -0.5`. The PoPE-specific path [4]
		passes `scale` to `flash_attn_with_pope`, and `scale` matches the default scaling in
		`Attend` [1].
	to_gates : nn.Linear
		Projection layer implemented with PyTorch `nn.Linear` [8] to produce one gate value for each
		head.
	to_out : nn.Sequential
		Output projection and dropout module implemented with PyTorch layers [8].
	to_qkv : nn.Linear
		Projection layer implemented with PyTorch `nn.Linear` [8] to produce concatenated query, key,
		and value features.

	See Also
	--------
	Attend
		Aggregate precomputed query, key, and value arrays.
	LinearAttention
		Project activations into the linear-attention variant.
	Transformer
		Stack `Attention` and `FeedForward` blocks with residual additions.

	References
	----------
	[1] hunterFormsBS.attend.Attend

	[2] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
		RoFormer: Enhanced Transformer with Rotary Position Embedding.
		https://arxiv.org/abs/2104.09864
	[3] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
		Decoupling The "What" and "Where" With Polar Coordinate Positional Embedding.
		https://arxiv.org/abs/2509.10534
	[4] lucidrains/PoPE-pytorch.
		https://github.com/lucidrains/PoPE-pytorch
	[5] hunterFormsBS.attend.Transformer

	[6] Lu, W.-T., Wang, J.-C., Kong, Q., and Hung, Y.-N. (2023).
		Music Source Separation with Band-Split RoPE Transformer.
		https://doi.org/10.48550/arXiv.2309.02612
	[7] Wang, J.-C., Lu, W.-T., and Won, M. (2023).
		Mel-Band RoFormer for Music Source Separation. https://arxiv.org/abs/2409.04702
	[8] PyTorch.
		https://context7.com/pytorch/pytorch
	"""
	def __init__(
		self,
		dim: int,
		dim_head: int = 64,
		dropout: float = 0.0,
		heads: int = 8,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
		scale: float | None = None,
		*,
		flash: bool = True,
		sage_attention: bool = False,
	) -> None:
		"""Set up an attention block for a chosen width and head layout.

		You can use `__init__` to decide how much feature capacity the block has, how many heads
		share that capacity, whether dropout is used, and which positional-encoding path the block
		should follow. `__init__` stores the resulting submodules so later calls to `forward` can
		reuse the same attention block.

		PyTorch
		-------
		submodule construction : implementation detail
			`__init__` computes `dim_inner = heads * dim_head`, stores `rotary_embed` and
			`pope_embed`, creates `Attend(flash=flash, dropout=dropout)` for the non-PoPE path [1],
			instantiates `RMSNorm(dim)`, and constructs `to_qkv = nn.Linear(dim, dim_inner * 3,
			bias=False)`, `to_gates = nn.Linear(dim, heads)`, and `to_out =
			nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout))`. `__init__`
			raises `ValueError` when both positional encoders are supplied.

		Parameters
		----------
		dim : int
			Input feature width before projection and output feature width after `to_out`.
		heads : int = 8
			Number of attention heads.
		dim_head : int = 64
			Feature width of each head before concatenation.
		dropout : float = 0.0
			Probability used by `Attend` [1] and the output dropout layer.
		rotary_embed : RotaryEmbedding | None = None
			Optional rotary position encoder [2]. `rotary_embed` is mutually exclusive with
			`pope_embed`.
		pope_embed : PoPE | None = None
			Optional polar coordinate position encoder [3][4]. `pope_embed` is mutually exclusive
			with `rotary_embed`.
		flash : bool = True
			Whether `Attend` [1] may use PyTorch SDPA (scaled dot-product attention) backends [5]
			when `pope_embed` is `None`.

		References
		----------
		[1] hunterFormsBS.attend.Attend

		[2] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
			RoFormer: Enhanced Transformer with Rotary Position Embedding.
			https://arxiv.org/abs/2104.09864
		[3] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
			Decoupling The "What" and "Where" With Polar Coordinate Positional Embedding.
			https://arxiv.org/abs/2509.10534
		[4] lucidrains/PoPE-pytorch.
			https://github.com/lucidrains/PoPE-pytorch
		[5] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		super().__init__()

		# Initialize `self`, primary
		self.heads: int = heads
		self.norm: RMSNorm = RMSNorm(dim)
		self.pope_embed: PoPE | None = pope_embed
		self.rotary_embed: RotaryEmbedding | None = rotary_embed
		self.scale: float = scale or dim_head**neg(0.5)

		# Initialize `self`, secondary
		self.attend: Attend = Attend(dropout=dropout, scale=self.scale, flash=flash, sage_attention=sage_attention)
		# "normal" `Attention`, not `LinearAttention`
		self.to_gates: nn.Linear = nn.Linear(dim, self.heads)

		# Compute internal values
		dim_inner: int = self.heads * dim_head

		# Initialize `self`, tertiary
		self.to_qkv: nn.Linear = nn.Linear(in_features=dim, out_features=dim_inner * 3, bias=False)
		self.to_out: nn.Sequential = nn.Sequential(nn.Linear(in_features=dim_inner, out_features=dim, bias=False), nn.Dropout(dropout))

	def forward(self, x: Tensor) -> Tensor:
		"""Compute gated multi-head attention output from activations `x`.

		You can use `forward` to normalize activations `x`, project activation `Tensor` `x` into
		query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v`, inject position through
		`rotary_embed` [2] or `pope_embed` [3][4], aggregate value `Tensor` `v` through
		`Attend.forward` [1] or the PoPE-specific flash-attention path [4], and return the gated,
		concatenated, projected result using the standard scaled dot-product attention mapping [5].
		`forward` preserves the batch axis and sequence axis of activation `Tensor` `x` and returns
		one updated feature vector for each input position.

		Parameters
		----------
		x : Tensor
			Input activation `Tensor` with shape `batch × sequence position × feature`.

		Returns
		-------
		out : Tensor
			Output activation `Tensor` with shape `batch × sequence position × feature`.

		Mathematics
		-----------
		gated multi-head attention : equation
		```
			Let H ≜ `self.heads`,  d ≜ `dim_head`,  α ≜ `self.scale` = d⁻¹ᐟ²,
				X ≜ `self.norm(x)`,  Q ≜ `q`,  K ≜ `k`,  V ≜ `v`,
				wₕᴳ ≜ `to_gates`.weight[h, :],  h ∈ {0, …, H−1},
				Wᴼ ≜ `to_out[0]`.weight

			S[b, h, i, j] = ⟨Q[b, h, i, :], K[b, h, j, :]⟩ · α
			A[b, h, i, j] = exp(S[b, h, i, j]) / ∑ₘ exp(S[b, h, i, m])
			O[b, h, i, :] = ∑ⱼ A[b, h, i, j] · V[b, h, j, :]
			G[b, i, h] = σ(⟨X[b, i, :], wₕᴳ⟩)
			Z[b, i, hd : (h+1)d] = G[b, i, h] · O[b, h, i, :]
			Y[b, i, :] = Z[b, i, :] Wᴼ

			where  G ≜ `gates`,  O ≜ `out`,  Y ≜ returned value
		```

		PyTorch
		-------
		path selection and head gating : implementation detail
			`forward` computes query `Tensor` `q`, key `Tensor` `k`, and value `Tensor` `v` from
			normalized `x`. When `pope_embed` is available, `forward` calls
			`flash_attn_with_pope` [4] with
			`pos_emb=self.pope_embed(x.shape[-2])` and `softmax_scale=self.scale`. When
			`pope_embed` is absent and `rotary_embed` is available, `forward` rotates query
			`Tensor` `q` and key `Tensor` `k` before `Attend.forward` [1]. When neither positional
			encoder is available, `forward` passes query `Tensor` `q`, key `Tensor` `k`, and value
			`Tensor` `v` directly to `Attend.forward` [1]. After the attention output is produced,
			`forward` multiplies each head output by `sigmoid(self.to_gates(X))`, concatenates the
			gated heads, and applies `to_out`.

		References
		----------
		[1] hunterFormsBS.attend.Attend.forward

		[2] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021).
			RoFormer: Enhanced Transformer with Rotary Position Embedding.
			https://arxiv.org/abs/2104.09864
		[3] Gopalakrishnan, A., Csordás, R., Schmidhuber, J., and Mozer, M. C. (2025).
			Decoupling The "What" and "Where" With Polar Coordinate Positional Embedding.
			https://arxiv.org/abs/2509.10534
		[4] lucidrains/PoPE-pytorch.
			https://github.com/lucidrains/PoPE-pytorch
		[5] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
			Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017).
			Attention Is All You Need.
			https://arxiv.org/abs/1706.03762
		"""
		x = self.norm(x)

		q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

		if exists(self.pope_embed):
			out: Tensor = flash_attn_with_pope(q, k, v, pos_emb=self.pope_embed(q.shape[-2]), softmax_scale=self.scale)
		elif exists(self.rotary_embed):
			q: Tensor = self.rotary_embed.rotate_queries_or_keys(q)
			k: Tensor = self.rotary_embed.rotate_queries_or_keys(k)
			out = self.attend(q, k, v)
		else:
			out = self.attend(q, k, v)

		# after attend
		# "normal" `Attention`, not `LinearAttention`
		gates: Tensor = self.to_gates(x)
		out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

		out = rearrange(out, 'b h n d -> b n (h d)')
		return self.to_out(out)

class FeedForward(Module):
	"""Transform activations with a position-wise expansion-and-projection block.

	You can use `FeedForward` after `Attention` [1] or `LinearAttention` [2] inside
	`Transformer` [3] when activation `Tensor` `x` at each sequence position should pass through
	the same nonlinear feature transform. `FeedForward` preserves the batch axis, sequence axis,
	and feature axis of activation `Tensor` `x` while expanding the feature width internally before
	projecting back to the original width. The shared BS-RoFormer [4] and Mel-Band RoFormer [5]
	stacks use one `FeedForward` block after every attention block.

	Attributes
	----------
	net : nn.Sequential
		Position-wise submodule sequence containing root-mean-square normalization, width expansion,
		`nn.GELU`, dropout, width projection, and output dropout [6].

	See Also
	--------
	Attention
		Provide the sequence-mixing sublayer that precedes `FeedForward` inside `Transformer`.
	LinearAttention
		Provide the optional cross-covariance attention sublayer used before `FeedForward`.
	Transformer
		Stack `Attention` or `LinearAttention` with `FeedForward` and residual additions.

	References
	----------
	[1] hunterFormsBS.attend.Attention

	[2] hunterFormsBS.attend.LinearAttention

	[3] hunterFormsBS.attend.Transformer

	[4] Lu, W.-T., Wang, J.-C., Kong, Q., and Hung, Y.-N. (2023).
		Music Source Separation with Band-Split RoPE Transformer.
		https://doi.org/10.48550/arXiv.2309.02612
	[5] Wang, J.-C., Lu, W.-T., and Won, M. (2023).
		Mel-Band RoFormer for Music Source Separation. https://arxiv.org/abs/2409.04702
	[6] PyTorch.
		https://context7.com/pytorch/pytorch
	"""
	def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0) -> None:
		"""Set up a position-wise feedforward block for feature width `dim`.

		You can use `__init__` to choose the input and output feature width `dim`, the internal width
		expansion factor `mult`, and the dropout probability `dropout` for the shared feedforward
		sublayer inside `Transformer` [1]. `__init__` stores the resulting normalization, linear,
		activation, and dropout layers in `net` for reuse by later calls to `forward`.

		Parameters
		----------
		dim : int
			Input feature width before expansion and output feature width after projection.
		mult : float = 4.0
			Multiplicative factor used to compute the hidden feature width `int(dim * mult)`.
		dropout : float = 0.0
			Dropout probability applied after the hidden activation and after the output projection.

		PyTorch
		-------
		submodule construction : implementation detail
			`__init__` computes `dim_inner = int(dim * mult)` and stores `net` as a six-stage
			`nn.Sequential` with root-mean-square normalization, `nn.Linear(dim, dim_inner)`,
			`nn.GELU`, `nn.Dropout(dropout)`, `nn.Linear(dim_inner, dim)`, and a second
			`nn.Dropout(dropout)` [2].

		References
		----------
		[1] hunterFormsBS.attend.Transformer

		[2] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		super().__init__()
		dim_inner: int = int(dim * mult)
		self.net: nn.Sequential = nn.Sequential(
			RMSNorm(dim), nn.Linear(dim, dim_inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_inner, dim), nn.Dropout(dropout)
		)

	def forward(self, x: Tensor) -> Tensor:
		"""Transform activations `x` with the position-wise feedforward block.

		You can use `forward` when `x` already has batch axis, sequence axis, and feature axis.
		`forward` applies the same nonlinear feature transform to every position independently,
		reusing the submodule sequence configured by `__init__`, and returns an activation `Tensor`
		with the same outer shape as `x`. `forward` supplies the position-wise branch used by
		`Transformer` [1].

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
			`forward` preserves the outer shape `B×N×D`. `forward` applies the same feedforward map
			to each position vector `x[b, n, :]` independently of every other position.

		References
		----------
		[1] hunterFormsBS.attend.Transformer
		"""
		return self.net(x)

class Transformer(Module):
	"""Refine activations with a repeated attention-and-feedforward stack.

	You can use `Transformer` as the shared sequence stack inside BS-RoFormer [4] and
	Mel-Band RoFormer [5]. `Transformer` takes activation `Tensor` `x`, repeats `depth` pairs of
	attention and feedforward sublayers, uses `Attention` [1] by default or `LinearAttention` [2]
	when `linear_attn=True`, and optionally normalizes the result. `Transformer` preserves the
	batch axis, sequence axis, and feature axis of activation `Tensor` `x`.

	Attributes
	----------
	layers : ModuleList
		Sequence of per-depth `[attention, feedforward]` pairs, where the attention entry is
		`Attention` [1] or `LinearAttention` [2] and the feedforward entry is `FeedForward` [3].
	norm : RMSNorm | nn.Identity
		Final output normalization module applied after the last residual block when `norm_output` is
		`True`, or `nn.Identity` otherwise.

	See Also
	--------
	Attention
		Provide the default sequence-position attention branch used by `Transformer`.
	LinearAttention
		Provide the optional cross-covariance attention branch used when `linear_attn=True`.
	FeedForward
		Provide the position-wise feature transform paired with each attention layer.

	References
	----------
	[1] hunterFormsBS.attend.Attention

	[2] hunterFormsBS.attend.LinearAttention

	[3] hunterFormsBS.attend.FeedForward

	[4] Lu, W.-T., Wang, J.-C., Kong, Q., and Hung, Y.-N. (2023).
		Music Source Separation with Band-Split RoPE Transformer.
		https://doi.org/10.48550/arXiv.2309.02612
	[5] Wang, J.-C., Lu, W.-T., and Won, M. (2023).
		Mel-Band RoFormer for Music Source Separation.
		https://arxiv.org/abs/2409.04702
	"""
	def __init__(
		self,
		*,
		attn_dropout: float = 0.0,
		depth: int,
		dim_head: int = 64,
		dim: int,
		ff_dropout: float = 0.0,
		ff_mult: float = 4,
		flash_attn: bool = True,
		heads: int = 8,
		linear_attn: bool = False,  # noqa: ARG002
		norm_output: bool = True,
		pope_embed: PoPE | None = None,
		rotary_embed: RotaryEmbedding | None = None,
		sage_attention: bool = False,
		scale: float = 8,
	) -> None:
		"""Set up a transformer stack for feature width `dim` and layer count `depth`.

		You can use `__init__` to choose the stack depth, model width, attention-head layout, dropout
		probabilities, attention backend, and positional-encoding path for a shared transformer
		block. `__init__` stores `depth` pairs of attention and feedforward sublayers in `layers` and
		configures optional output normalization in `norm`.

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
			Dropout probability passed to each `FeedForward` block [3].
		ff_mult : float = 4
			Hidden-width expansion factor passed to each `FeedForward` block [3].
		flash_attn : bool = True
			Whether each attention sublayer may use PyTorch SDPA (scaled dot-product attention)
			backends [4].
		heads : int = 8
			Number of attention heads in each attention sublayer.
		linear_attn : bool = False
			Whether to build each attention sublayer with `LinearAttention` [2] instead of
			`Attention` [1].
		norm_output : bool = True
			Whether to apply final output normalization after the last residual block.
		pope_embed : PoPE | None = None
			Optional polar coordinate position encoder passed to `Attention` [1] when
			`linear_attn=False`. `pope_embed` is ignored when `linear_attn=True`.
		rotary_embed : RotaryEmbedding | None = None
			Optional rotary positional encoder passed to `Attention` [1] when `linear_attn=False`.
			`rotary_embed` is ignored when `linear_attn=True`.

		PyTorch
		-------
		layer construction : implementation detail
			`__init__` builds `depth` `ModuleList` pairs. Each pair stores one attention sublayer and
			one feedforward sublayer. When `linear_attn=True`, the attention sublayer is
			`LinearAttention(**attentionKwargs)` [2]. Otherwise, `__init__` creates `Attention` [1]
			and passes `rotary_embed` plus `pope_embed` to each `Attention`. The paired feedforward
			sublayer is `FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)` [3]. `norm` becomes
			`RMSNorm(dim)` when `norm_output` is `True` and `nn.Identity()` otherwise [4].

		References
		----------
		[1] hunterFormsBS.attend.Attention

		[2] hunterFormsBS.attend.LinearAttention

		[3] hunterFormsBS.attend.FeedForward

		[4] PyTorch.
			https://context7.com/pytorch/pytorch
		"""
		super().__init__()

		attentionKwargs = KwargsOfAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
				flash=flash_attn, scale=scale, pope_embed=pope_embed, rotary_embed=rotary_embed, sage_attention=sage_attention)

		self.layers = ModuleList(
			ModuleList([Attention(**attentionKwargs), FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)])
				for _deep in loops(depth)
		)

		self.norm: RMSNorm | nn.Identity = RMSNorm(dim) if norm_output else nn.Identity()

	def forward(self, x: Tensor) -> Tensor:
		"""Transform activations `x` through the residual stack.

		You can use `forward` when `x` already has batch axis, sequence axis, and feature axis.
		`forward` applies each stored `Attention` [1] or `LinearAttention` [2] sublayer and each
		stored `FeedForward` [3] sublayer with residual additions and then applies `norm`. `forward`
		returns an activation `Tensor` with the same outer shape as `x`.

		Parameters
		----------
		x : Tensor
			Input activation `Tensor` with shape `batch × sequence position × feature`.

		Returns
		-------
		transformed_x : Tensor
			Output activation `Tensor` with shape `batch × sequence position × feature`.

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

			where  Y ≜ `transformed_x`
		```

		References
		----------
		[1] hunterFormsBS.attend.Attention

		[2] hunterFormsBS.attend.LinearAttention

		[3] hunterFormsBS.attend.FeedForward
		"""
		for sherpa in self.layers:
			attn: Attention = cast('Attention', cast('ModuleList', sherpa)[0])
			ff: FeedForward = cast('FeedForward', cast('ModuleList', sherpa)[1])
			x = attn(x) + x
			x = ff(x) + x

		return self.norm(x)
