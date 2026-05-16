# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: D100, ARG002, D101, D102
from __future__ import annotations

from einops import pack, rearrange, reduce, repeat, unpack  # pyright: ignore[reportUnknownVariableType]
from functools import partial
from hunterFormsBS.attend_experimental import Transformer
from hunterFormsBS.bandSplit import BandSplit, DEFAULT_FREQS_PER_BANDS, lossComputation, mask_filter_bank_mel_band_default, MaskEstimator
from hunterFormsBS.theTypes import ComputeLoss, KwargsSTFT, KwargsTransformer
from hunterMakesPy import raiseIfNone
from hyper_connections import get_init_and_expand_reduce_stream_functions  # NOTE There is a newer version.
from more_itertools import loops
from operator import mul
from PoPE_pytorch import PoPE
from rotary_embedding_torch import RotaryEmbedding
from torch import nn, tensor, Tensor
from torch.nn import Module, ModuleList
from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportUnknownVariableType]
from torch_einops_kit import default, exists
from torch_einops_kit.einops import pack_one, unpack_one
from torch_einops_kit.scaleValues import RMSNorm
from typing import cast, TYPE_CHECKING
from Z0Z_tools import halfsineTensor
import torch

if TYPE_CHECKING:
	from collections.abc import Callable

class MelBandRoformer(Module):
	def __init__(
		self,
		dim: int,
		*,
		attn_dropout: float = 0.0,
		depth: int,
		dim_freqs_in: int = 1025,
		dim_head: int = 64,
		ff_dropout: float = 0.0,
		final_norm: bool | None = None,
		flash_attn: bool = True,
		freq_transformer_depth: int = 2,
		freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
		heads: int = 8,
		linear_transformer_depth: int = 0,
		mask_estimator_depth: int | None = None,
		mask_filter_bank: Tensor | None = None,
		match_input_audio_length: bool = True,
		mc_hyper_conn_sinkhorn_iters: int | None = None,
		mlp_expansion_factor: int = 4,
		multi_stft_hop_size: int = 147,
		multi_stft_normalized: bool = False,
		multi_stft_resolution_loss_weight: float = 1.0,
		multi_stft_resolutions_window_sizes: tuple[int, ...] = (4096, 2048, 1024, 512, 256),
		multi_stft_window_fn: Callable[..., Tensor] = halfsineTensor,
		norm_output: bool | None = None,
		num_bands: int | None = None,
		num_residual_fracs: int | None = None,
		num_residual_streams: int = 1,
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
		use_value_residual_learning: bool = False,
		zero_dc: bool | None = None,
	) -> None:
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
			if norm_output is None:
				norm_output = True
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
			dim=dim, ff_dropout=ff_dropout, flash_attn=flash_attn, heads=heads,
			norm_output=raiseIfNone(norm_output, f'I received {norm_output = }, but I need a type `bool` value or a "truthy" value.'),
		)

		self.num_residual_streams = num_residual_streams
		_init_hyper_conn, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(self.num_residual_streams, disable=self.num_residual_streams == 1)

		self.layers: ModuleList = ModuleList([])
		for layer_index in range(depth):
			if use_value_residual_learning:
				is_first = layer_index == 0
			else:
				is_first = True

			self.layers.append(nn.ModuleList([
				Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, pope_embed=time_pope_embed,
					add_value_residual=not is_first,
					num_residual_streams=self.num_residual_streams,
					**transformer_kwargs)
				, Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, pope_embed=freq_pope_embed,
					add_value_residual=not is_first,
					num_residual_streams=self.num_residual_streams,
					**transformer_kwargs)
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
		time_v_residual = None
		freq_v_residual = None

		if self.num_residual_streams != 1:
			x = self.expand_stream(x)

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
				x, next_time_v_residual = checkpoint(time_transformer, x, time_v_residual, use_reentrant=False)  # pyright: ignore[reportGeneralTypeIssues]
			else:
				x, next_time_v_residual = time_transformer(x, value_residual=time_v_residual)
			time_v_residual = default(time_v_residual, next_time_v_residual)

			(x,) = unpack(x, ps, '* t d')
			x = rearrange(x, 'b f t d -> b t f d')
			x, ps = pack([x], '* f d')

			if self.use_torch_checkpoint:
				x, next_freq_v_residual = checkpoint(freq_transformer, x, freq_v_residual, use_reentrant=False)  # pyright: ignore[reportGeneralTypeIssues]
			else:
				x, next_freq_v_residual = freq_transformer(x, value_residual=freq_v_residual)
			freq_v_residual = default(freq_v_residual, next_freq_v_residual)

			(x,) = unpack(x, ps, '* f d')

			if self.skip_connection:
				store[i] = x

		if self.num_residual_streams != 1:
			x = self.reduce_stream(x)

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
