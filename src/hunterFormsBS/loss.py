"""Compute multi-resolution training loss for waveform source separation.

You can use this module to train a separator that reconstructs waveform audio and matches target
spectrogram structure at multiple STFT window sizes. `lossComputation` combines a waveform-domain
mean absolute error term with a multi-resolution complex-STFT term to produce the training
objective. The loss family follows the BS-RoFormer [1] and Mel-RoFormer [2] formulations and is
consumed by `hunterFormsBS.bandSplitRotator.BandSplitRotator` [3].

Contents
--------
Functions
	lossComputation
		Compute waveform MAE and multi-resolution complex-STFT MAE for selected stems and return one
		total loss or an expanded breakdown.

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
from hunterFormsBS.theTypes import ParametersComputeLoss, ParametersSTFT
from torch import Tensor, tensor
from typing import Literal, overload
import torch
import torch.nn.functional as F

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
