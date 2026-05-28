from __future__ import annotations

from hunterFormsBS.transform import FeedForward
from tests.dataSamples.tensors import (
	tensorFeedForwardAlfa, tensorFeedForwardBeta, tensorFeedForwardExpectedAlfaDim5, tensorFeedForwardExpectedAlfaDim13,
	tensorFeedForwardExpectedBetaDim5, tensorFeedForwardExpectedBetaDim13)
from torch import nn, Tensor
import pytest
import torch

@pytest.mark.parametrize('ff_dropout', [0.0, 0.25], ids=['drop0', 'drop25'])
@pytest.mark.parametrize('ff_mult', [2.0, 4.0], ids=['mult2', 'mult4'])
@pytest.mark.parametrize(
	('dim', 'x', 'expected'),
	[
		pytest.param(5, tensorFeedForwardAlfa, tensorFeedForwardExpectedAlfaDim5, id='dim5-alfa'),
		pytest.param(5, tensorFeedForwardBeta, tensorFeedForwardExpectedBetaDim5, id='dim5-beta'),
		pytest.param(13, tensorFeedForwardAlfa, tensorFeedForwardExpectedAlfaDim13, id='dim13-alfa'),
		pytest.param(13, tensorFeedForwardBeta, tensorFeedForwardExpectedBetaDim13, id='dim13-beta'),
	],
)
def test_FeedForward(dim: int, ff_mult: float, ff_dropout: float, x: Tensor, expected: Tensor) -> None:
	feedForward: FeedForward = FeedForward(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout)

	with torch.no_grad():
		firstLinearModule = feedForward.net[1]
		secondLinearModule = feedForward.net[4]
		assert isinstance(firstLinearModule, nn.Linear), (
			f'FeedForward stored {type(firstLinearModule).__name__} at index 1, expected Linear for {dim=}, {ff_mult=}, {ff_dropout=}, {tuple(x.shape)=}.'
		)
		assert isinstance(secondLinearModule, nn.Linear), (
			f'FeedForward stored {type(secondLinearModule).__name__} at index 4, expected Linear for {dim=}, {ff_mult=}, {ff_dropout=}, {tuple(x.shape)=}.'
		)
		firstLinear: nn.Linear = firstLinearModule
		secondLinear: nn.Linear = secondLinearModule
		firstLinear.weight.zero_()
		firstLinear.bias.zero_()
		secondLinear.weight.zero_()
		secondLinear.bias.zero_()

	actual: Tensor = feedForward(x[..., 0:dim])
	assert actual.shape == expected.shape, (
		f'FeedForward.forward returned shape {tuple(actual.shape)}, expected {tuple(expected.shape)} for {dim=}, {ff_mult=}, {ff_dropout=}, {tuple(x.shape)=}.'
	)
	assert torch.equal(actual, expected), (
		f'FeedForward.forward returned {actual}, expected {expected} for {dim=}, {ff_mult=}, {ff_dropout=}, {tuple(x.shape)=}.'
	)

@pytest.mark.parametrize(
	('dim', 'ff_mult', 'ff_dropout', 'x', 'expected'),
	[
		pytest.param(5, 2.0, -0.5, tensorFeedForwardAlfa, ValueError, id='dropout-negative-alfa'),
		pytest.param(5, 2.0, 1.5, tensorFeedForwardBeta, ValueError, id='dropout-too-large-beta'),
		pytest.param(-5, 2.0, 0.0, tensorFeedForwardAlfa, RuntimeError, id='dim-negative-alfa'),
		pytest.param(5, -2.0, 0.0, tensorFeedForwardBeta, RuntimeError, id='mult-negative-beta'),
	],
)
def test_FeedForwardError(dim: int, ff_mult: float, ff_dropout: float, x: Tensor, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		FeedForward(dim=dim, ff_mult=ff_mult, ff_dropout=ff_dropout)
