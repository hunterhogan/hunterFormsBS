from __future__ import annotations

from hunterFormsBS.mask import MLP
from torch import nn
import pytest

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
