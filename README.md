# hunterFormsBS

A flexible frequency-band splitter for music source separation, organized around a single separator family that can express BS-style, mel-style, and custom layouts.

Instead of treating `BSRoformer` and `MelBandRoformer` as separate architectures, this package treats them as different band-layout configurations of one core design centered on `BandSplitRotator`.

[![pip install hunterFormsBS](https://img.shields.io/badge/pip_install-hunterFormsBS-gray.svg?labelColor=blue)](https://pypi.org/project/hunterFormsBS/)
[![uv add hunterFormsBS](https://img.shields.io/badge/uv_add-hunterFormsBS-gray.svg?labelColor=blue)](https://pypi.org/project/hunterFormsBS/)

The codebase is implemented in PyTorch, fully typed (`py.typed`), and designed for modular reuse so research ideas (for example PoPE or custom filter banks) can be integrated without splitting into parallel implementations.

## Quick fix: size mismatch when loading a checkpoint

If loading a `BSRoFormer` checkpoint raises a size-mismatch error, check `mask_estimator_depth` in the configuration.

Some upstream configurations effectively used `mask_estimator_depth=1` even when set to `2` because a later subtraction was applied. This package removes that subtraction, so the direct equivalent is:

- set `mask_estimator_depth=1`

Updating that value resolves the most common mismatch quickly.

## Why this architecture helps in practice

- **Forward-looking architecture:** A single model family makes it easier to adopt new ideas, such as PoPE or custom band-split definitions, while keeping interfaces aligned with established ecosystems.
- **Universal configuration:** Configurable backward compatibility with existing standards.
- **Rich tooling & Ecosystem:** The package provides strong typing (`py.typed`), modular APIs, and rich docstrings focusing on usage, literature citations, and migration paths.

## Easy to migrate

Transitioning from other standard implementations is straightforward because **most identifiers are exactly the same** and the data flow is highly similar.

If you're changing from an existing codebase, you can use the **transition modules**: simply keep using the `BSRoformer` and `MelBandRoformer` namespaces and APIs as a bridge, unify your other classes, and then switch to `BandSplitRotator` when you're ready.

## What is unified here

The key design idea is that the difference between the BS-style front end and the mel-band front end
is treated as a band-layout problem, not as a reason to maintain two unrelated model families.

- `hunterFormsBS.bandSplitRotator.BandSplitRotator` is the new universal entry point.
- `hunterFormsBS.bs_roformer.BSRoformer` and `hunterFormsBS.mel_band_roformer.MelBandRoformer` serve as transition modules, keeping familiar APIs, upstream names, and defaults.
- `hunterFormsBS.bandSplit.BandSplit`, `hunterFormsBS.bandSplit.MaskEstimator`, and
  `hunterFormsBS.attend.Transformer` hold the reusable typed building blocks shared across those entry
  points.

At the band level, the model only needs a band-membership map, called `mask_filter_bank` in the
codebase. You can think of that map as a Boolean matrix

$$
F \in \{0, 1\}^{B \times N_f}
$$

where $B$ is the number of bands and $N_f$ is the number of STFT frequency bins.

- In a non-overlapping BS-style layout, each frequency bin belongs to exactly one band, so

  $$
  \forall f,\; \sum_b F_{b,f} = 1.
  $$

- In an overlapping mel-style layout, some frequency bins belong to more than one band, so

  $$
  \exists f \text{ such that } \sum_b F_{b,f} > 1.
  $$

When bands overlap, the reconstructed mask for a frequency bin is averaged across the contributing
bands:

$$
\hat{M}_{f,t} = \frac{1}{S_f} \sum_{b : F_{b,f} = 1} \hat{M}^{(b)}_{f,t},
\qquad
S_f = \sum_b F_{b,f}.
$$

That is why this package makes it easy to move between overlapping and non-overlapping bands, and to
change how bands are distributed across the frequency axis. The architectural difference lives in the
filter bank, not in two separate theories of the model.

## Which entry point you should use

| Use this                                          | When                                                                                                           | Why                                                                              |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `hunterFormsBS.BandSplitRotator`                  | You are starting new work or want one separator that can cover BS-style, mel-style, and custom band layouts.   | This is the unified model entry point.                                           |
| `hunterFormsBS.bs_roformer.BSRoformer`            | You want the familiar non-overlapping BS-style interface or a close comparison with upstream BS-RoFormer code. | The constructor keeps BS-oriented defaults and compatibility fields.             |
| `hunterFormsBS.mel_band_roformer.MelBandRoformer` | You want the familiar mel-band interface or a close comparison with upstream mel-band code.                    | The constructor keeps mel-oriented defaults and automatic mel-band construction. |
| `hunterFormsBS.*_experimental`                    | You are testing research ideas such as value residual learning or hyper-connections.                           | These modules are exploratory and intentionally separate from the stable path.   |

## Package map

| Module                            | Main symbols                                                                                                                                                                                                 | Purpose                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `hunterFormsBS.__init__`          | `BandSplitRotator`, `BandSplit`, `MaskEstimator`, `Transformer`, `lossComputation`, `DEFAULT_FREQS_PER_BANDS`, `ComputeLoss`, `FlashAttentionConfig`, `KwargsOfAttention`, `KwargsSTFT`, `KwargsTransformer` | Public top-level namespace for the stable typed API.                                                 |
| `hunterFormsBS.bandSplitRotator`  | `BandSplitRotator`                                                                                                                                                                                           | Unified separator that can build BS-style, mel-style, or custom band layouts from one model family.  |
| `hunterFormsBS.bs_roformer`       | `BSRoformer`                                                                                                                                                                                                 | Stable compatibility module for the non-overlapping BS-style variant.                                |
| `hunterFormsBS.mel_band_roformer` | `MelBandRoformer`                                                                                                                                                                                            | Stable compatibility module for the overlapping mel-band variant.                                    |
| `hunterFormsBS.bandSplit`         | `BandSplit`, `MaskEstimator`, `MLP`, `lossComputation`, `DEFAULT_FREQS_PER_BANDS`                                                                                                                            | Shared band projection, mask-estimation heads, BS-style default partition, and training-loss helper. |
| `hunterFormsBS.attend`            | `Attend`, `Attention`, `FeedForward`, `LinearAttention`, `Transformer`                                                                                                                                       | Stable attention, feedforward, linear-attention, and transformer building blocks.                    |
| `hunterFormsBS.theTypes`          | `ComputeLoss`, `FlashAttentionConfig`, `KwargsOfAttention`, `KwargsSTFT`, `KwargsTransformer`                                                                                                                | Typed configuration records used across the package.                                                 |

## Experimental module map

| Module                                         | Main symbols                                         | Purpose                                                                                     |
| ---------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `hunterFormsBS.attend_experimental`            | experimental `Attention`, experimental `Transformer` | Research-oriented attention blocks with value-residual mixing and hyper-connection support. |
| `hunterFormsBS.bs_roformer_experimental`       | experimental `BSRoformer`                            | Experimental BS-style separator that uses the experimental attention stack.                 |
| `hunterFormsBS.mel_band_roformer_experimental` | experimental `MelBandRoformer`                       | Experimental mel-band separator that uses the experimental attention stack.                 |

## Architecture in one sentence

The stable separator path is

`raw audio → STFT → band gathering → BandSplit → hierarchical attention → MaskEstimator → mask`

followed by overlap-aware mask averaging when needed, complex masking in the STFT domain, and inverse
STFT reconstruction back to waveform audio.

## Top-level exports

The top-level package namespace currently re-exports the stable shared pieces that new users most
often need:

- `BandSplitRotator`

The compatibility classes are intentionally available from their own modules so that imports can stay
explicit during comparisons with upstream repos.

## Reference materials

### Gaussian Error Linear Units (GELUs)

- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/hendrycks2016gaussian.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/1606.08415)
- eprint: [arXiv.1606.08415](https://arXiv.org/abs/1606.08415)
- Implementations:
  - [hendrycks/GELUs](https://github.com/hendrycks/GELUs)
  - [torch.nn.GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)

### Language Modeling with Gated Convolutional Networks

- Common name: GLU (Gated Linear Units)
- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/pmlr-v70-dauphin17a.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/1612.08083)
- Proceedings: [proceedings.mlr.press](http://proceedings.mlr.press/v70/dauphin17a/dauphin17a.pdf)

### Attention Is All You Need

(^Which is why there are no other papers on this list.)

- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/NIPS2017_3f5ee243.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/1706.03762)
- Proceedings: [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

### Root Mean Square Layer Normalization

- Common name: RMSNorm
- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/NEURIPS2019_1e8a1942.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/1910.07467)
- Proceedings: [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Paper.pdf)
- Implementations:
  - [bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm)
    - [context7](https://context7.com/bzhangGo/rmsnorm)
  - [hunterhogan/torch_einops_kit](https://github.com/hunterhogan/torch_einops_kit).scaleValues.RMSNorm
    - [context7](https://context7.com/hunterhogan/torch_einops_kit)

### RoFormer: Enhanced Transformer with Rotary Position Embedding

- Common name: RoPE
- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/SU2024127063.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/2104.09864)
- DOI: [10.1016/j.neucom.2023.127063](https://doi.org/10.1016/j.neucom.2023.127063)
- Free pre-print: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Implementations:
  - [lucidrains/rotary-embedding-torch](https://github.com/lucidrains/rotary-embedding-torch)
    - [context7](https://context7.com/lucidrains/rotary-embedding-torch)
  - [hunterhogan/rotary-embedding-torch](https://github.com/hunterhogan/rotary-embedding-torch)

### XCiT: Cross-Covariance Image Transformers

- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/NEURIPS2021_a655fbe4.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/2106.09681)
- Proceedings: [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2021/file/a655fbe4b8d7439994aa37ddad80de56-Paper.pdf)

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/NEURIPS2022_67d57c32.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/2205.14135)
- Proceedings: [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)
- Implementations:
  - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
    - [context7](https://context7.com/dao-ailab/flash-attention)

### Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation

- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/rogozhnikov2022einops.bib)
- Proceedings: [ICLR 2022 Conference](https://openreview.net/pdf?id=oapKSVM2bcj)
- Implementations:
  - [arogozhnikov/einops](https://github.com/arogozhnikov/einops)
    - [Official documentation](https://einops.rocks/)
    - [context7](https://context7.com/arogozhnikov/einops)

### Music Source Separation with Band-Split RoPE Transformer

- Common name: BS-RoFormer
- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/10446843.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/2309.02612)
- DOI: [10.1109/ICASSP48485.2024.10446843](https://doi.org/10.1109/ICASSP48485.2024.10446843)
- Free pre-print: [arXiv:2309.02612](https://arxiv.org/pdf/2309.02612)
- Implementations:
  - [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
    - [context7](https://context7.com/lucidrains/bs-roformer)

### Mel-RoFormer for Vocal Separation and Vocal Melody Transcription

- Common name: MelBand-RoFormer
- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/ju_chiang_wang_2024_14877371.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/2409.04702)
- Proceedings: [10.5281/zenodo.14877371](https://doi.org/10.5281/zenodo.14877371)
- Implementations:
  - [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
    - [context7](https://context7.com/lucidrains/bs-roformer)

### Value Residual Learning

- Common name: ResFormer
- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/zhou-etal-2025-value.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/2410.17897)
- Proceedings: [10.18653/v1/2025.acl-long.1375](https://aclanthology.org/2025.acl-long.1375.pdf)
- Implementations:
  - [Zcchill/Value-Residual-Learning](https://github.com/Zcchill/Value-Residual-Learning)
    - [context7](https://context7.com/zcchill/value-residual-learning)

### Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings

- Common name: PoPE
- [BibTeX citation.](https://github.com/hunterhogan/hunterFormsBS/blob/main/citations/gopalakrishnan2025decoupling.bib) [TeX Source with precise formulas for AI agents.](https://arxiv.org/src/2509.10534)
- eprint: [arXiv.2509.10534](https://arxiv.org/pdf/2509.10534)
- Implementations:
  - [lucidrains/PoPE-pytorch](https://github.com/lucidrains/PoPE-pytorch)
    - [context7](https://context7.com/lucidrains/pope-pytorch)
  - [hunterhogan/PoPE-pytorch](https://github.com/hunterhogan/PoPE-pytorch)
    - [context7](https://context7.com/hunterhogan/pope-pytorch/admin)

<!-- Incorporate this? [Review and rejection by ICLR 2026 Conference](https://openreview.net/forum?id=kf2mzS6xfk) -->

### Packages and documentation

- [pytorch/pytorch](https://github.com/pytorch/pytorch)
  - [official documentation](https://docs.pytorch.org/docs/)
  - [context7](https://context7.com/pytorch/pytorch)
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
  - [context7](https://context7.com/zfturbo/music-source-separation-training)
- [lucidrains/torch-einops-utils](https://github.com/lucidrains/torch-einops-utils)
  - [context7](https://context7.com/lucidrains/torch-einops-utils)
- [hunterhogan/torch_einops_kit](https://github.com/hunterhogan/torch_einops_kit)
  - [context7](https://context7.com/hunterhogan/torch_einops_kit)

## My recovery

[![Static Badge](https://img.shields.io/badge/2011_August-Homeless_since-blue?style=flat)](https://HunterThinks.com/support)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC3Gx7kz61009NbhpRtPP7tw)](https://www.youtube.com/@HunterHogan)

[![CC-BY-NC-4.0](https://raw.githubusercontent.com/hunterhogan/hunterFormsBS/refs/heads/main/.github/CC-BY-NC-4.0.png)](https://creativecommons.org/licenses/by-nc/4.0/)
