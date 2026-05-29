from __future__ import annotations

from hunterFormsBS.loss import lossComputation
from math import isclose
from tests.dataSamples.loss import multi_stft
from tests.dataSamples.tensors import AIgarbage_recon_audio, AIgarbage_target
from torch import Tensor
from typing import cast, TYPE_CHECKING
import contextlib
import pytest
import torch

if TYPE_CHECKING:
	from hunterFormsBS.theTypes import ParametersComputeLoss

@pytest.mark.parametrize(
	('recon_audio', 'target', 'stem_ids', 'multi_stft', 'return_loss_breakdown', 'expected'),
	[
		pytest.param(AIgarbage_recon_audio[:, [0]], AIgarbage_target, [0], multi_stft, False, 0.127830, id='var0-1stems-brkFalse'),
		pytest.param(AIgarbage_recon_audio[:, [0]], AIgarbage_target, [0], multi_stft, True, (0.127830, (0.004918, 0.122911)), id='var0-1stems-brkTrue'),
		pytest.param(AIgarbage_recon_audio[:, [0, 1]], AIgarbage_target, [0, 1], multi_stft, False, 0.217115, id='var0-2stems-brkFalse'),
		pytest.param(AIgarbage_recon_audio[:, [0, 1]], AIgarbage_target, [0, 1], multi_stft, True, (0.217115, (0.006378, 0.210737)), id='var0-2stems-brkTrue'),
		pytest.param(AIgarbage_recon_audio[:, [0, 1, 2, 3]], AIgarbage_target, [0, 1, 2, 3], multi_stft, False, 0.353176, id='var0-4stems-brkFalse'),
		pytest.param(AIgarbage_recon_audio[:, [0, 1, 2, 3]], AIgarbage_target, [0, 1, 2, 3], multi_stft, True, (0.353176, (0.014863, 0.338313)), id='var0-4stems-brkTrue'),
		pytest.param(AIgarbage_recon_audio[:, [0, 1, 2, 3, 4, 5]], AIgarbage_target, [0, 1, 2, 3, 4, 5], multi_stft, False, 0.235451, id='var0-6stems-brkFalse'),
		pytest.param(AIgarbage_recon_audio[:, [0, 1, 2, 3, 4, 5]], AIgarbage_target, [0, 1, 2, 3, 4, 5], multi_stft, True, (0.235451, (0.009909, 0.225542)), id='var0-6stems-brkTrue'),
	],
)
def test_lossComputation(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ParametersComputeLoss, return_loss_breakdown: bool, expected: float | tuple[float, tuple[float, float]]) -> None:
	with torch.no_grad():
		actual = lossComputation(recon_audio, target, stem_ids, multi_stft, return_loss_breakdown=return_loss_breakdown)

	if return_loss_breakdown:
		assert isinstance(actual, tuple), (
			f"lossComputation returned {type(actual).__name__}, expected tuple when {return_loss_breakdown=}."
		)
		total_loss, (loss, multi_stft_resolution_loss) = actual
		total_lossExpected, (lossExpected, multi_stft_resolution_lossExpected) = cast('tuple[float, tuple[float, float]]', expected)

		assert isclose(total_loss.item(), total_lossExpected, rel_tol=1e-4, abs_tol=1e-4), (
			f"lossComputation total loss returned {total_loss.item()}, expected {total_lossExpected}."
		)
		assert isclose(loss.item(), lossExpected, rel_tol=1e-4, abs_tol=1e-4), (
			f"lossComputation MAE returned {loss.item()}, expected {lossExpected}."
		)
		assert isclose(multi_stft_resolution_loss.item(), multi_stft_resolution_lossExpected, rel_tol=1e-4, abs_tol=1e-4), (
			f"lossComputation STFT loss returned {multi_stft_resolution_loss.item()}, expected {multi_stft_resolution_lossExpected}."
		)
	else:
		assert isinstance(actual, Tensor), (
			f"lossComputation returned {type(actual).__name__}, expected Tensor when {return_loss_breakdown=}."
		)
		assert isclose(actual.item(), cast('float', expected), rel_tol=1e-4, abs_tol=1e-4), (
			f"lossComputation returned {actual.item()}, expected {expected}."
		)

@pytest.mark.parametrize(
	('recon_audio', 'target', 'stem_ids', 'multi_stft', 'expected_error', 'expected_warnings'),
	[
		pytest.param(AIgarbage_recon_audio[:, [0]], AIgarbage_target, [99], multi_stft, IndexError, None, id='stem_id_out_of_bounds'),
		pytest.param(AIgarbage_recon_audio[:, [0, 1]], AIgarbage_target, [0], multi_stft, RuntimeError, UserWarning, id='shape_mismatch_recon_target'),
	]
)
def test_lossComputationError(recon_audio: Tensor, target: Tensor, stem_ids: list[int], multi_stft: ParametersComputeLoss, expected_error: type[Exception], expected_warnings: type[Warning] | None) -> None:
	warns_ctx = pytest.warns(expected_warnings) if expected_warnings is not None else contextlib.nullcontext()
	with warns_ctx, pytest.raises(expected_error), torch.no_grad():
		lossComputation(recon_audio=recon_audio, target=target, stem_ids=stem_ids, multi_stft=multi_stft, return_loss_breakdown=False)
