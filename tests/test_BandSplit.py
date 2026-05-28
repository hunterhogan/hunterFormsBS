from __future__ import annotations

from hunterFormsBS.bands import BandSplit
from tests.dataSamples.bands import dim_inputsAlfa, dim_inputsBeta
from tests.dataSamples.tensors import (
	tensorBandFeaturesAlfa, tensorBandFeaturesAlfaExpected, tensorBandFeaturesBeta, tensorBandFeaturesBetaExpected, tensorWaveformAlfa)
from torch import nn, Tensor
from typing import TYPE_CHECKING
import pytest
import torch

if TYPE_CHECKING:
	from collections.abc import Sequence

@pytest.mark.parametrize(
	('dim', 'dim_inputs', 'x', 'expected'),
	[
		pytest.param(1, dim_inputsAlfa, tensorBandFeaturesAlfa, tensorBandFeaturesAlfaExpected, id='alfa-forward'),
		pytest.param(1, dim_inputsBeta, tensorBandFeaturesBeta, tensorBandFeaturesBetaExpected, id='beta-forward'),
	],
)
def test_BandSplit(dim: int, dim_inputs: Sequence[int], x: Tensor, expected: Tensor) -> None:
	bandSplit: BandSplit = BandSplit(dim=dim, dim_inputs=dim_inputs)

	tensorBandBiases: Tensor = expected.select(dim=0, index=0)
	assert len(bandSplit.to_features) == tensorBandBiases.shape[0], (
		f'BandSplit configured {len(bandSplit.to_features)} projection modules, expected {tensorBandBiases.shape[0]} for {tuple(x.shape)=}.'
	)

	with torch.no_grad():
		for bandIndex, (toFeature, tensorExpectedBandBias) in enumerate(zip(bandSplit.to_features, tensorBandBiases, strict=True)):
			assert isinstance(toFeature, nn.Sequential), (
				f'BandSplit projection module {bandIndex} stored {type(toFeature).__name__}, expected Sequential.'
			)
			linearProjection = toFeature.get_submodule('1')
			assert isinstance(linearProjection, nn.Linear), (
				f'BandSplit projection module {bandIndex} stored {type(linearProjection).__name__}, expected Linear.'
			)
			assert linearProjection.bias is not None, f'BandSplit projection module {bandIndex} did not expose a bias parameter.'
			linearProjection.weight.zero_()
			linearProjection.bias.copy_(tensorExpectedBandBias)

	tensorActualOutput: Tensor = bandSplit.forward(x)
	assert tensorActualOutput.shape == expected.shape, (
		f'BandSplit.forward returned shape {tuple(tensorActualOutput.shape)}, expected {tuple(expected.shape)} for {tuple(x.shape)=}.'
	)
	assert torch.equal(tensorActualOutput, expected), (
		f'BandSplit.forward returned {tensorActualOutput}, expected {expected} for {tuple(x.shape)=}.'
	)

@pytest.mark.parametrize(
	('dim', 'dim_inputs', 'x', 'expected'),
	[
		pytest.param(1, dim_inputsAlfa, tensorBandFeaturesBeta, RuntimeError, id='beta-shape-mismatch'),
		pytest.param(1, dim_inputsAlfa, tensorWaveformAlfa, RuntimeError, id='waveform-shape-mismatch'),
		pytest.param(1, dim_inputsBeta, tensorBandFeaturesAlfa, RuntimeError, id='alfa-shape-mismatch'),
		pytest.param(1, dim_inputsBeta, tensorWaveformAlfa, RuntimeError, id='waveform-shape-mismatch'),
	],
)
def test_BandSplit_exception(dim: int, dim_inputs: Sequence[int], x: Tensor, expected: type[Exception]) -> None:
	bandSplit: BandSplit = BandSplit(dim=dim, dim_inputs=dim_inputs)
	with pytest.raises(expected):
		bandSplit.forward(x)
