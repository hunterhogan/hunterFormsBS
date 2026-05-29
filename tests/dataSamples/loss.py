from __future__ import annotations

from hunterFormsBS.theTypes import ParametersComputeLoss
from Z0Z_tools import halfsineTensor

multi_stft = ParametersComputeLoss(n_fft=2048, hop_length=147, normalized=False, loss_weight=1.0, window_sizes=(4096, 2048, 1024, 512, 256), window_fn=halfsineTensor)
