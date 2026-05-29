from __future__ import annotations

from hunterFormsBS.mask import MaskEstimator, MLP
from tests.dataSamples.tensors import (
	tensorFeedForwardAlfa, tensorMaskEstimatorExpectedAlfa, tensorMaskEstimatorExpectedBeta, tensorMaskEstimatorInputAlfa,
	tensorMaskEstimatorInputBeta)
from torch import nn, Tensor
from typing import TYPE_CHECKING
import pytest
import torch

if TYPE_CHECKING:
	from collections.abc import Sequence

@pytest.mark.parametrize(
	('dim_inputs', 'x', 'expected'),
	[
		pytest.param([8, 12, 16], tensorMaskEstimatorInputAlfa, tensorMaskEstimatorExpectedAlfa, id='bands3-expected36-alfa'),
		pytest.param([7, 9, 11, 13], tensorMaskEstimatorInputBeta, tensorMaskEstimatorExpectedBeta, id='bands4-expected40-beta'),
	],
)
@pytest.mark.parametrize('activation', [nn.Tanh], ids=['activation-tanh'])
@pytest.mark.parametrize('dim', [5, 13], ids=['dim5', 'dim13'])
@pytest.mark.parametrize('depth', [0, 2], ids=['depth0', 'depth2'])
@pytest.mark.parametrize('mlp_expansion_factor', [2, 4], ids=['exp2', 'exp4'])
def test_MaskEstimator(dim: int, dim_inputs: Sequence[int], depth: int, mlp_expansion_factor: int, activation: type[nn.Module], x: Tensor, expected: Tensor) -> None:
	maskEstimator: MaskEstimator = MaskEstimator(
		dim=dim
		, dim_inputs=dim_inputs
		, depth=depth
		, mlp_expansion_factor=mlp_expansion_factor
		, activation=activation
	)

	assert len(maskEstimator.to_freqs) == len(dim_inputs), (
		f'MaskEstimator configured {len(maskEstimator.to_freqs)} projection modules, expected {len(dim_inputs)} for {dim=}, {depth=}, {mlp_expansion_factor=}, {tuple(x.shape)=}.'
	)

	xPrepared: Tensor = x[..., 0:len(dim_inputs), 0:dim]

	with torch.no_grad():
		bandOffset: int = 0
		for bandIndex, (toFreq, bandWidth) in enumerate(zip(maskEstimator.to_freqs, dim_inputs, strict=True)):
			assert isinstance(toFreq, nn.Sequential), (
				f'MaskEstimator projection module {bandIndex} stored {type(toFreq).__name__}, expected Sequential.'
			)
			mlpModule = toFreq[0]
			assert isinstance(mlpModule, nn.Sequential), (
				f'MaskEstimator projection module {bandIndex} stored {type(mlpModule).__name__} at index 0, expected Sequential.'
			)

			linearModules: list[nn.Linear] = [module for module in mlpModule if isinstance(module, nn.Linear)]
			assert len(linearModules) > 0, (
				f'MaskEstimator projection module {bandIndex} did not include a Linear module for {dim=}, {depth=}, {mlp_expansion_factor=}.'
			)

			for linearModule in linearModules:
				assert linearModule.bias is not None, f'MaskEstimator Linear module in projection {bandIndex} did not expose a bias parameter.'
				linearModule.weight.zero_()
				linearModule.bias.zero_()

			tensorExpectedBandOutput: Tensor = expected[0, 0, bandOffset:bandOffset + bandWidth]
			tensorGateBias: Tensor = torch.cat((tensorExpectedBandOutput * 2.0, torch.zeros_like(tensorExpectedBandOutput)), dim=0)
			finalLinear: nn.Linear = linearModules[-1]
			assert finalLinear.bias is not None, f'MaskEstimator final Linear module in projection {bandIndex} did not expose a bias parameter.'
			finalLinear.bias.copy_(tensorGateBias)
			bandOffset += bandWidth

	actual: Tensor = maskEstimator.forward(xPrepared)
	assert actual.shape == expected.shape, (
		f'MaskEstimator.forward returned shape {tuple(actual.shape)}, expected {tuple(expected.shape)} for {dim=}, {depth=}, {mlp_expansion_factor=}, {tuple(xPrepared.shape)=}.'
	)
	assert torch.equal(actual, expected), (
		f'MaskEstimator.forward returned {actual}, expected {expected} for {dim=}, {depth=}, {mlp_expansion_factor=}, {tuple(xPrepared.shape)=}.'
	)

@pytest.mark.parametrize(
	('dim_inputs', 'x', 'expected'),
	[
		pytest.param([8, 12, 16], tensorMaskEstimatorInputAlfa, ValueError, id='bands3-input-has4-bands'),
		pytest.param([7, 9, 11, 13], tensorFeedForwardAlfa, ValueError, id='bands4-input-has2-bands'),
	],
)
@pytest.mark.parametrize('activation', [nn.Tanh], ids=['activation-tanh'])
@pytest.mark.parametrize('dim', [5, 13], ids=['dim5', 'dim13'])
@pytest.mark.parametrize('depth', [0, 2], ids=['depth0', 'depth2'])
@pytest.mark.parametrize('mlp_expansion_factor', [2, 4], ids=['exp2', 'exp4'])
def test_MaskEstimatorError(dim: int, dim_inputs: Sequence[int], depth: int, mlp_expansion_factor: int, activation: type[nn.Module], x: Tensor, expected: type[Exception]) -> None:
	maskEstimator: MaskEstimator = MaskEstimator(
		dim=dim
		, dim_inputs=dim_inputs
		, depth=depth
		, mlp_expansion_factor=mlp_expansion_factor
		, activation=activation,
	)

	xPrepared: Tensor = x[..., 0:dim]
	with pytest.raises(expected):
		maskEstimator.forward(xPrepared)

@pytest.mark.parametrize(
	('dim_in', 'dim_out', 'dim_hidden', 'depth', 'activation', 'expected'),
	[
		pytest.param(5, 8, None, 0, nn.Tanh, nn.Sequential(nn.Linear(5, 8)), id='dim_in5-dim_out8-dim_hiddenNone-depth0'),
		pytest.param(5, 8, None, 2, nn.Tanh, nn.Sequential(nn.Linear(5, 5), nn.Tanh(), nn.Linear(5, 5), nn.Tanh(), nn.Linear(5, 8)), id='dim_in5-dim_out8-dim_hiddenNone-depth2'),
		pytest.param(5, 8, 34, 0, nn.Tanh, nn.Sequential(nn.Linear(5, 8)), id='dim_in5-dim_out8-dim_hidden34-depth0'),
		pytest.param(5, 8, 34, 2, nn.Tanh, nn.Sequential(nn.Linear(5, 34), nn.Tanh(), nn.Linear(34, 34), nn.Tanh(), nn.Linear(34, 8)), id='dim_in5-dim_out8-dim_hidden34-depth2'),
		pytest.param(13, 8, None, 0, nn.Tanh, nn.Sequential(nn.Linear(13, 8)), id='dim_in13-dim_out8-dim_hiddenNone-depth0'),
		pytest.param(13, 8, None, 2, nn.Tanh, nn.Sequential(nn.Linear(13, 13), nn.Tanh(), nn.Linear(13, 13), nn.Tanh(), nn.Linear(13, 8)), id='dim_in13-dim_out8-dim_hiddenNone-depth2'),
		pytest.param(13, 8, 34, 0, nn.Tanh, nn.Sequential(nn.Linear(13, 8)), id='dim_in13-dim_out8-dim_hidden34-depth0'),
		pytest.param(13, 8, 34, 2, nn.Tanh, nn.Sequential(nn.Linear(13, 34), nn.Tanh(), nn.Linear(34, 34), nn.Tanh(), nn.Linear(34, 8)), id='dim_in13-dim_out8-dim_hidden34-depth2'),
		pytest.param(5, 21, None, 0, nn.Tanh, nn.Sequential(nn.Linear(5, 21)), id='dim_in5-dim_out21-dim_hiddenNone-depth0'),
		pytest.param(5, 21, None, 2, nn.Tanh, nn.Sequential(nn.Linear(5, 5), nn.Tanh(), nn.Linear(5, 5), nn.Tanh(), nn.Linear(5, 21)), id='dim_in5-dim_out21-dim_hiddenNone-depth2'),
		pytest.param(5, 21, 34, 0, nn.Tanh, nn.Sequential(nn.Linear(5, 21)), id='dim_in5-dim_out21-dim_hidden34-depth0'),
		pytest.param(5, 21, 34, 2, nn.Tanh, nn.Sequential(nn.Linear(5, 34), nn.Tanh(), nn.Linear(34, 34), nn.Tanh(), nn.Linear(34, 21)), id='dim_in5-dim_out21-dim_hidden34-depth2'),
		pytest.param(13, 21, None, 0, nn.Tanh, nn.Sequential(nn.Linear(13, 21)), id='dim_in13-dim_out21-dim_hiddenNone-depth0'),
		pytest.param(13, 21, None, 2, nn.Tanh, nn.Sequential(nn.Linear(13, 13), nn.Tanh(), nn.Linear(13, 13), nn.Tanh(), nn.Linear(13, 21)), id='dim_in13-dim_out21-dim_hiddenNone-depth2'),
		pytest.param(13, 21, 34, 0, nn.Tanh, nn.Sequential(nn.Linear(13, 21)), id='dim_in13-dim_out21-dim_hidden34-depth0'),
		pytest.param(13, 21, 34, 2, nn.Tanh, nn.Sequential(nn.Linear(13, 34), nn.Tanh(), nn.Linear(34, 34), nn.Tanh(), nn.Linear(34, 21)), id='dim_in13-dim_out21-dim_hidden34-depth2'),
	],
)
def test_MLP(dim_in: int, dim_out: int, dim_hidden: int | None, depth: int, activation: type[nn.Module], expected: nn.Sequential) -> None:
	actual: nn.Sequential = MLP(dim_in=dim_in, dim_out=dim_out, dim_hidden=dim_hidden, depth=depth, activation=activation)
	assert str(actual) == str(expected), (
		f'MLP returned {actual}, expected {expected} for dim_in={dim_in}, dim_out={dim_out}, dim_hidden={dim_hidden}, depth={depth}.'
	)

@pytest.mark.parametrize(
	('dim_in', 'dim_out', 'dim_hidden', 'depth', 'activation', 'expected'),
	[
		pytest.param(-5, 8, None, 2, nn.Tanh, RuntimeError, id='dim_in-5-dim_out8-dim_hiddenNone-depth2'),
		pytest.param(5, -8, None, 2, nn.Tanh, RuntimeError, id='dim_in5-dim_out-8-dim_hiddenNone-depth2'),
		pytest.param(5, 8, -34, 2, nn.Tanh, RuntimeError, id='dim_in5-dim_out8-dim_hidden-34-depth2'),
	],
)
def test_MLPError(dim_in: int, dim_out: int, dim_hidden: int | None, depth: int, activation: type[nn.Module], expected: type[Exception]) -> None:
	with pytest.raises(expected):
		MLP(dim_in=dim_in, dim_out=dim_out, dim_hidden=dim_hidden, depth=depth, activation=activation)
