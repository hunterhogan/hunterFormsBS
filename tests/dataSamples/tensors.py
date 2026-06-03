from __future__ import annotations

from hunterHearsPy import readAudioFile
from torch import from_numpy, tensor  # pyright: ignore[reportUnknownVariableType]
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
	from torch import Tensor

# https://en.wikipedia.org/wiki/NATO_phonetic_alphabet

tensorBandFeaturesAlfa: Tensor = tensor(dtype=torch.float32, data=[[2.0, 3.0, 5.0, 7.0] * 1025, [11.0, 13.0, 17.0, 19.0] * 1025])
tensorBandFeaturesAlfaExpected: Tensor = tensor(dtype=torch.float32, data=[[[2.0], [3.0], [5.0], [7.0], [11.0], [13.0], [17.0], [19.0], [23.0], [29.0], [31.0], [37.0], [41.0], [43.0], [47.0], [53.0], [59.0], [61.0], [67.0], [71.0], [73.0], [79.0], [83.0], [89.0], [97.0], [101.0], [103.0], [107.0], [109.0], [113.0], [127.0], [131.0], [137.0], [139.0], [149.0], [151.0], [157.0], [163.0], [167.0], [173.0], [179.0], [181.0], [191.0], [193.0], [197.0], [199.0], [211.0], [223.0], [227.0], [229.0], [233.0], [239.0], [241.0], [251.0], [257.0], [263.0], [269.0], [271.0], [277.0], [281.0], [283.0], [293.0]]] * 2)
tensorBandFeaturesBeta: Tensor = tensor(dtype=torch.float32, data=[[23.0, 29.0, 31.0, 37.0] * 1979, [41.0, 43.0, 47.0, 53.0] * 1979, [59.0, 61.0, 67.0, 71.0] * 1979])
tensorBandFeaturesBetaExpected: Tensor = tensor(dtype=torch.float32, data=[[[307.0], [311.0], [313.0], [317.0], [331.0], [337.0], [347.0], [349.0], [353.0], [359.0], [367.0], [373.0], [379.0], [383.0], [389.0], [397.0], [401.0], [409.0], [419.0], [421.0], [431.0], [433.0], [439.0], [443.0], [449.0], [457.0], [461.0], [463.0], [467.0], [479.0], [487.0], [491.0], [499.0], [503.0], [509.0], [521.0], [523.0], [541.0], [547.0], [557.0], [563.0], [569.0], [571.0], [577.0], [587.0], [593.0], [599.0], [601.0], [607.0], [613.0], [617.0], [619.0], [631.0], [641.0], [643.0], [647.0], [653.0], [659.0], [661.0], [673.0]]] * 3)
tensorFeedForwardAlfa: Tensor = tensor(dtype=torch.float32, data=[[[2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0], [43.0, 47.0, 53.0, 59.0, 61.0, 67.0, 71.0, 73.0, 79.0, 83.0, 89.0, 97.0, 101.0]], [[103.0, 107.0, 109.0, 113.0, 127.0, 131.0, 137.0, 139.0, 149.0, 151.0, 157.0, 163.0, 167.0], [173.0, 179.0, 181.0, 191.0, 193.0, 197.0, 199.0, 211.0, 223.0, 227.0, 229.0, 233.0, 239.0]]])
tensorFeedForwardAlfaExpanded: Tensor = torch.cat((tensorFeedForwardAlfa, tensorFeedForwardAlfa[:, 0:1, :]), dim=1)
tensorFeedForwardBeta: Tensor = tensor(dtype=torch.float32, data=[[[241.0, 251.0, 257.0, 263.0, 269.0, 271.0, 277.0, 281.0, 283.0, 293.0, 307.0, 311.0, 313.0], [317.0, 331.0, 337.0, 347.0, 349.0, 353.0, 359.0, 367.0, 373.0, 379.0, 383.0, 389.0, 397.0], [401.0, 409.0, 419.0, 421.0, 431.0, 433.0, 439.0, 443.0, 449.0, 457.0, 461.0, 463.0, 467.0]], [[479.0, 487.0, 491.0, 499.0, 503.0, 509.0, 521.0, 523.0, 541.0, 547.0, 557.0, 563.0, 569.0], [571.0, 577.0, 587.0, 593.0, 599.0, 601.0, 607.0, 613.0, 617.0, 619.0, 631.0, 641.0, 643.0], [647.0, 653.0, 659.0, 661.0, 673.0, 677.0, 683.0, 691.0, 701.0, 709.0, 719.0, 727.0, 733.0]]])
tensorFeedForwardExpectedAlfaDim5: Tensor = tensor(dtype=torch.float32, data=[[[0.0] * 5, [0.0] * 5], [[0.0] * 5, [0.0] * 5]])
tensorFeedForwardExpectedAlfaDim13: Tensor = tensor(dtype=torch.float32, data=[[[0.0] * 13, [0.0] * 13], [[0.0] * 13, [0.0] * 13]])
tensorFeedForwardExpectedBetaDim5: Tensor = tensor(dtype=torch.float32, data=[[[0.0] * 5, [0.0] * 5, [0.0] * 5], [[0.0] * 5, [0.0] * 5, [0.0] * 5]])
tensorFeedForwardExpectedBetaDim13: Tensor = tensor(dtype=torch.float32, data=[[[0.0] * 13, [0.0] * 13, [0.0] * 13], [[0.0] * 13, [0.0] * 13, [0.0] * 13]])
tensorMaskEstimatorExpectedAlfa: Tensor = tensor(dtype=torch.float32, data=[[[2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0, 59.0, 61.0, 67.0, 71.0, 73.0, 79.0, 83.0, 89.0, 97.0, 101.0, 103.0, 107.0, 109.0, 113.0, 127.0, 131.0, 137.0, 139.0, 149.0, 151.0]] * 3] * 2)
tensorMaskEstimatorExpectedBeta: Tensor = tensor(dtype=torch.float32, data=[[[157.0, 163.0, 167.0, 173.0, 179.0, 181.0, 191.0, 193.0, 197.0, 199.0, 211.0, 223.0, 227.0, 229.0, 233.0, 239.0, 241.0, 251.0, 257.0, 263.0, 269.0, 271.0, 277.0, 281.0, 283.0, 293.0, 307.0, 311.0, 313.0, 317.0, 331.0, 337.0, 347.0, 349.0, 353.0, 359.0, 367.0, 373.0, 379.0, 383.0]] * 3] * 2)
tensorMaskEstimatorInputAlfa: Tensor = torch.stack((tensorFeedForwardAlfaExpanded, tensorFeedForwardBeta, tensorFeedForwardAlfaExpanded + 1000.0, tensorFeedForwardBeta + 1000.0), dim=2)
tensorMaskEstimatorInputBeta: Tensor = torch.stack((tensorFeedForwardBeta + 2000.0, tensorFeedForwardAlfaExpanded + 2000.0, tensorFeedForwardBeta + 4000.0, tensorFeedForwardAlfaExpanded + 4000.0), dim=2)
tensorWaveformAlfa: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/LUFS20_44100_ch2_birdsPink.wav'))
tensorWaveformBeta: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/bass.wav'))
tensorWaveformCharlie: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/drums.wav'))
tensorWaveformDelta: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/mixture.wav'))
tensorWaveformEcho: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/other.wav'))
tensorWaveformFoxtrot: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/vocals.wav'))
tensorWaveformGolf: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/bassBad.wav'))
tensorWaveformHotel: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/bassGood.wav'))
tensorWaveformIndia: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/drumsBad.wav'))
tensorWaveformJuliet: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/drumsGood.wav'))
tensorWaveformKilo: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/otherBad.wav'))
tensorWaveformLima: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/otherGood.wav'))
tensorWaveformMike: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/vocalsBad.wav'))
tensorWaveformNovember: torch.Tensor = from_numpy(readAudioFile('tests/dataSamples/SpeakSoftly_BrokenMan60sec/vocalsGood.wav'))






# NOTE This moronic crap is not what I want, but I'm tired of fighting the LLM. But clearly worse than the AI code are the unintelligible identifiers from the human-made code: 'recon_audio'? srsly?
AIgarbage_target: torch.Tensor = torch.stack((torch.stack((tensorWaveformHotel, tensorWaveformJuliet, tensorWaveformLima, tensorWaveformNovember, tensorWaveformBeta, tensorWaveformCharlie))[..., :44100], torch.stack((tensorWaveformHotel, tensorWaveformJuliet, tensorWaveformLima, tensorWaveformNovember, tensorWaveformBeta, tensorWaveformCharlie))[..., 44100:88200]), dim=0)
AIgarbage_recon_audio: torch.Tensor = torch.stack((torch.stack((tensorWaveformGolf, tensorWaveformIndia, tensorWaveformKilo, tensorWaveformMike, tensorWaveformBeta, tensorWaveformCharlie))[..., :44100], torch.stack((tensorWaveformGolf, tensorWaveformIndia, tensorWaveformFoxtrot, tensorWaveformEcho, tensorWaveformBeta, tensorWaveformCharlie))[..., 44100:88200]), dim=0)
