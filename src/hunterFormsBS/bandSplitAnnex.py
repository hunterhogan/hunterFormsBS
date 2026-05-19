# ruff: noqa: D100 D101, D102, D103, E741
from __future__ import annotations

from more_itertools import loops
from torch import nn, Tensor
from typing import cast, overload, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
	from collections.abc import Callable

class Conv(nn.Module):
	def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, act: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bias)
		self.bn = nn.InstanceNorm2d(c2, affine=norm_affine, eps=norm_eps)
		self.act: nn.Module = activation() if act else nn.Identity()

	def forward(self, x: Tensor) -> Tensor:
		return self.act(self.bn(self.conv(x)))

@overload
def autopad(k: int, p: int | None = None) -> int:...
@overload
def autopad(k: list[int], p: list[int] | None = None) -> list[int]:...
def autopad(k: int | list[int], p: int | list[int] | None = None) -> int | list[int]:
	if p is None:
		p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
	return p

class DSConv(nn.Module):
	def __init__(self, c1: int, c2: int, k: int = 3, s: int | tuple[int, int] = 1, p: int | None = None
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, act: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		self.dwconv = nn.Conv2d(c1, c1, k, s, autopad(k, p), groups=c1, bias=bias)
		self.pwconv = nn.Conv2d(c1, c2, 1, 1, 0, bias=bias)
		self.bn = nn.InstanceNorm2d(c2, affine=norm_affine, eps=norm_eps)
		self.act: nn.Module = activation() if act else nn.Identity()

	def forward(self, x: Tensor) -> Tensor:
		return self.act(self.bn(self.pwconv(self.dwconv(x))))

class DS_Bottleneck(nn.Module):
	def __init__(self, c1: int, c2: int, k: int = 3
			, activation: type[nn.Module] = nn.SiLU
			, norm_eps: float = 1e-8
			, *
			, shortcut: bool = True
			, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		c_: int = c1
		self.dsconv1: DSConv = DSConv(c1, c_, k=3, s=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.dsconv2: DSConv = DSConv(c_, c2, k=k, s=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.shortcut: bool = shortcut and c1 == c2

	def forward(self, x: Tensor) -> Tensor:
		return x + self.dsconv2(self.dsconv1(x)) if self.shortcut else self.dsconv2(self.dsconv1(x))

class DS_C3k(nn.Module):
	def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5, activation: type[nn.Module] = nn.SiLU, norm_eps: float = 1e-8, *, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		c_ = int(c2 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv2: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv3: Conv = Conv(2 * c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.m = nn.Sequential(
			*[
				DS_Bottleneck(c_, c_, k=k, shortcut=True, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
				for _ in range(n)
			]
		)

	def forward(self, x: Tensor) -> Tensor:
		return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class DS_C3k2(nn.Module):
	def __init__(self, c1: int, c2: int, n: int = 1, k: int = 3, e: float = 0.5, activation: type[nn.Module] = nn.SiLU, norm_eps: float = 1e-8, *, norm_affine: bool = True, bias: bool = False) -> None:
		super().__init__()
		c_ = int(c2 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.m: DS_C3k = DS_C3k(c_, c_, n=n, k=k, e=1.0, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)
		self.cv2: Conv = Conv(c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=bias)

	def forward(self, x: Tensor) -> Tensor:
		x_ = self.cv1(x)
		x_ = self.m(x_)
		return self.cv2(x_)

class AdaptiveHyperedgeGeneration(nn.Module):
	def __init__(self, in_channels: int, num_hyperedges: int, num_heads: int = 8, *, linear_bias: bool = False) -> None:
		super().__init__()
		self.num_hyperedges: int = num_hyperedges
		self.num_heads: int = num_heads
		self.head_dim: int = in_channels // num_heads

		self.global_proto: nn.Parameter = nn.Parameter(torch.randn(num_hyperedges, in_channels))

		self.context_mapper: nn.Linear = nn.Linear(2 * in_channels, num_hyperedges * in_channels, bias=linear_bias)

		self.query_proj: nn.Linear = nn.Linear(in_channels, in_channels, bias=linear_bias)

		self.scale: float = self.head_dim**-0.5

	def forward(self, x: Tensor) -> Tensor:
		B, N, C = x.shape

		f_avg: Tensor = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
		f_max: Tensor = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
		f_ctx: Tensor = torch.cat((f_avg, f_max), dim=1)

		delta_P = self.context_mapper(f_ctx).view(B, self.num_hyperedges, C)
		P = self.global_proto.unsqueeze(0) + delta_P

		z = self.query_proj(x)

		z = z.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

		P = P.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

		sim = (z @ P) * self.scale

		s_bar = sim.mean(dim=1)

		A: Tensor = F.softmax(s_bar.permute(0, 2, 1), dim=-1)

		return A

class HypergraphConvolution(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, activation: type[nn.Module] = nn.SiLU, *, linear_bias: bool = False) -> None:
		super().__init__()
		self.W_e = nn.Linear(in_channels, in_channels, bias=linear_bias)
		self.W_v = nn.Linear(in_channels, out_channels, bias=linear_bias)
		self.act: nn.Module = activation()

	def forward(self, x: Tensor, A: Tensor) -> Tensor:
		f_m: Tensor = torch.bmm(A, x)
		f_m = self.act(self.W_e(f_m))

		x_out: Tensor = torch.bmm(A.transpose(1, 2), f_m)
		x_out = self.act(self.W_v(x_out))

		return x + x_out

class AdaptiveHypergraphComputation(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, num_hyperedges: int = 8, num_heads: int = 8, activation: type[nn.Module] = nn.SiLU, *, linear_bias: bool = False) -> None:
		super().__init__()
		self.adaptive_hyperedge_gen: AdaptiveHyperedgeGeneration = AdaptiveHyperedgeGeneration(in_channels, num_hyperedges, num_heads, linear_bias=linear_bias)
		self.hypergraph_conv: HypergraphConvolution = HypergraphConvolution(in_channels, out_channels, activation=activation, linear_bias=linear_bias)

	def forward(self, x: Tensor) -> Tensor:
		B, _C, H, W = x.shape
		x_flat: Tensor = x.flatten(2).permute(0, 2, 1)

		A = self.adaptive_hyperedge_gen(x_flat)

		x_out_flat = self.hypergraph_conv(x_flat, A)

		return x_out_flat.permute(0, 2, 1).view(B, -1, H, W)

class C3AH(nn.Module):
	def __init__(
		self,
		c1: int,
		c2: int,
		num_hyperedges: int = 8,
		num_heads: int = 8,
		e: float = 0.5,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()
		c_ = int(c1 * e)
		self.cv1: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.cv2: Conv = Conv(c1, c_, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.ahc: AdaptiveHypergraphComputation = AdaptiveHypergraphComputation(c_, c_, num_hyperedges, num_heads, activation=activation, linear_bias=linear_bias)
		self.cv3: Conv = Conv(2 * c_, c2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)

	def forward(self, x: Tensor) -> Tensor:
		x_lateral = self.cv1(x)
		x_ahc = self.ahc(self.cv2(x))
		return self.cv3(torch.cat((x_ahc, x_lateral), dim=1))

class HyperACE(nn.Module):
	def __init__(
		self,
		in_channels: list[int],
		out_channels: int,
		num_hyperedges: int = 8,
		num_heads: int = 8,
		k: int = 2,
		l: int = 1,
		c_h: float = 0.5,
		c_l: float = 0.25,
		c3ah_expansion: float = 1.0,
		low_order_depth: int = 1,
		low_order_kernel: int = 3,
		low_order_expansion: float = 1.0,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()

		c2, c3, c4, c5 = in_channels
		c_mid: int = c4

		self.fuse_conv: Conv = Conv(
			c2 + c3 + c4 + c5, c_mid, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)

		self.c_h = int(c_mid * c_h)
		self.c_l = int(c_mid * c_l)
		self.c_s: int = c_mid - self.c_h - self.c_l
		if self.c_s <= 0:
			message: str = (
				f"I computed `{self.c_s = }`, indicative of a channel split problem, from `{c_mid = }`, `{c_h = }`, `{c_l = }`, "
				f"`{self.c_h = }`, and `{self.c_l = }`, but I need `self.c_s` to be greater than 0."
			)
			raise ValueError(message)

		self.high_order_branch = nn.ModuleList(
			[
				C3AH(
					self.c_h,
					self.c_h,
					num_hyperedges,
					num_heads,
					e=c3ah_expansion,
					activation=activation,
					norm_eps=norm_eps,
					norm_affine=norm_affine,
					conv_bias=conv_bias,
					linear_bias=linear_bias,
				)
				for _index in loops(k)
			]
		)
		self.high_order_fuse: Conv = Conv(
			self.c_h * k, self.c_h, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)

		self.low_order_branch = nn.Sequential(
			*[
				DS_C3k(
					self.c_l,
					self.c_l,
					n=low_order_depth,
					k=low_order_kernel,
					e=low_order_expansion,
					activation=activation,
					norm_eps=norm_eps,
					norm_affine=norm_affine,
					bias=conv_bias,
				)
				for _ in range(l)
			]
		)

		self.final_fuse: Conv = Conv(
			self.c_h + self.c_l + self.c_s,
			out_channels,
			1,
			1,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)

	def forward(self, x: list[Tensor]) -> Tensor:
		B2, B3, B4, B5 = x

		_B, _C, H4, W4 = B4.shape

		B2_resized: Tensor = F.interpolate(B2, size=(H4, W4), mode='bilinear', align_corners=False)
		B3_resized: Tensor = F.interpolate(B3, size=(H4, W4), mode='bilinear', align_corners=False)
		B5_resized: Tensor = F.interpolate(B5, size=(H4, W4), mode='bilinear', align_corners=False)

		x_b = self.fuse_conv(torch.cat((B2_resized, B3_resized, B4, B5_resized), dim=1))

		x_h, x_l, x_s = torch.split(x_b, [self.c_h, self.c_l, self.c_s], dim=1)

		x_h_outs: list[Tensor] = [m(x_h) for m in self.high_order_branch]
		x_h_fused = self.high_order_fuse(torch.cat(x_h_outs, dim=1))

		x_l_out = self.low_order_branch(x_l)

		return self.final_fuse(torch.cat((x_h_fused, x_l_out, x_s), dim=1))

class GatedFusion(nn.Module):
	def __init__(self, in_channels: int) -> None:
		super().__init__()
		self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

	def forward(self, f_in: Tensor, h: Tensor) -> Tensor:
		if f_in.shape[1] != h.shape[1]:
			message: str = (
				f"I received `{f_in.shape = }` and `{h.shape = }`, but I need the number of channels to match, so "
				f"`{f_in.shape[1] = }` to equal `{h.shape[1] = }`."
			)
			raise ValueError(message)
		return f_in + self.gamma * h

class Backbone(nn.Module):
	def __init__(
		self,
		in_channels: int = 256,
		base_channels: int = 64,
		base_depth: int = 3,
		channels: tuple[int, int, int, int, int] | None = None,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
	) -> None:
		super().__init__()
		if channels is None:
			c2: int = base_channels
			c3 = 256
			c4 = 384
			c5 = 512
			c6 = 768
		else:
			c2, c3, c4, c5, c6 = channels

		self.stem: DSConv = DSConv(
			in_channels, c2, k=3, s=(2, 1), p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)

		self.p2 = nn.Sequential(
			DSConv(c2, c3, k=3, s=(2, 1), p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
			DS_C3k2(c3, c3, n=base_depth, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
		)

		self.p3 = nn.Sequential(
			DSConv(c3, c4, k=3, s=(2, 1), p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
			DS_C3k2(c4, c4, n=base_depth * 2, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
		)

		self.p4 = nn.Sequential(
			DSConv(c4, c5, k=3, s=2, p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
			DS_C3k2(c5, c5, n=base_depth * 2, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
		)

		self.p5 = nn.Sequential(
			DSConv(c5, c6, k=3, s=2, p=1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
			DS_C3k2(c6, c6, n=base_depth, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias),
		)

		self.out_channels: list[int] = [c3, c4, c5, c6]

	def forward(self, x: Tensor) -> list[Tensor]:
		x = self.stem(x)
		x2 = self.p2(x)
		x3 = self.p3(x2)
		x4 = self.p4(x3)
		x5 = self.p5(x4)
		return [x2, x3, x4, x5]

class Decoder(nn.Module):
	def __init__(
		self,
		encoder_channels: list[int],
		hyperace_out_c: int,
		decoder_channels: list[int],
		block_depth: int = 1,
		block_kernel: int = 3,
		block_expansion: float = 0.5,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
	) -> None:
		super().__init__()
		c_p2, c_p3, c_p4, c_p5 = encoder_channels
		c_d2, c_d3, c_d4, c_d5 = decoder_channels

		self.h_to_d5: Conv = Conv(hyperace_out_c, c_d5, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.h_to_d4: Conv = Conv(hyperace_out_c, c_d4, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.h_to_d3: Conv = Conv(hyperace_out_c, c_d3, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.h_to_d2: Conv = Conv(hyperace_out_c, c_d2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)

		self.fusion_d5: GatedFusion = GatedFusion(c_d5)
		self.fusion_d4: GatedFusion = GatedFusion(c_d4)
		self.fusion_d3: GatedFusion = GatedFusion(c_d3)
		self.fusion_d2: GatedFusion = GatedFusion(c_d2)

		self.skip_p5: Conv = Conv(c_p5, c_d5, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.skip_p4: Conv = Conv(c_p4, c_d4, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.skip_p3: Conv = Conv(c_p3, c_d3, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)
		self.skip_p2: Conv = Conv(c_p2, c_d2, 1, 1, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias)

		self.up_d5: DS_C3k2 = DS_C3k2(
			c_d5,
			c_d4,
			n=block_depth,
			k=block_kernel,
			e=block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)
		self.up_d4: DS_C3k2 = DS_C3k2(
			c_d4,
			c_d3,
			n=block_depth,
			k=block_kernel,
			e=block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)
		self.up_d3: DS_C3k2 = DS_C3k2(
			c_d3,
			c_d2,
			n=block_depth,
			k=block_kernel,
			e=block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)

		self.final_d2: DS_C3k2 = DS_C3k2(
			c_d2,
			c_d2,
			n=block_depth,
			k=block_kernel,
			e=block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			bias=conv_bias,
		)

	def forward(self, enc_feats: list[Tensor], h_ace: Tensor) -> Tensor:
		p2, p3, p4, p5 = enc_feats

		d5 = self.skip_p5(p5)
		h_d5 = self.h_to_d5(F.interpolate(h_ace, size=d5.shape[2:], mode='bilinear'))
		d5 = self.fusion_d5(d5, h_d5)

		d5_up: Tensor = F.interpolate(d5, size=p4.shape[2:], mode='bilinear')
		d4_skip = self.skip_p4(p4)
		d4 = self.up_d5(d5_up) + d4_skip

		h_d4 = self.h_to_d4(F.interpolate(h_ace, size=d4.shape[2:], mode='bilinear'))
		d4 = self.fusion_d4(d4, h_d4)

		d4_up: Tensor = F.interpolate(d4, size=p3.shape[2:], mode='bilinear')
		d3_skip = self.skip_p3(p3)
		d3 = self.up_d4(d4_up) + d3_skip

		h_d3 = self.h_to_d3(F.interpolate(h_ace, size=d3.shape[2:], mode='bilinear'))
		d3 = self.fusion_d3(d3, h_d3)

		d3_up: Tensor = F.interpolate(d3, size=p2.shape[2:], mode='bilinear')
		d2_skip = self.skip_p2(p2)
		d2 = self.up_d3(d3_up) + d2_skip

		h_d2 = self.h_to_d2(F.interpolate(h_ace, size=d2.shape[2:], mode='bilinear'))
		d2 = self.fusion_d2(d2, h_d2)

		return self.final_d2(d2)

class TFC_TDF(nn.Module):
	def __init__(self, in_c: int, c: int, l: int, f: int, bn: int = 4, activation: type[nn.Module] = nn.SiLU, norm_eps: float = 1e-8, *, norm_affine: bool = True, conv_bias: bool = False, linear_bias: bool = False) -> None:
		super().__init__()

		self.blocks = nn.ModuleList()
		for _index in loops(l):
			block = nn.Module()

			block.tfc1 = nn.Sequential(
				nn.InstanceNorm2d(in_c, affine=norm_affine, eps=norm_eps), activation(), nn.Conv2d(in_c, c, 3, 1, 1, bias=conv_bias)
			)
			block.tdf = nn.Sequential(
				nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps),
				activation(),
				nn.Linear(f, f // bn, bias=linear_bias),
				nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps),
				activation(),
				nn.Linear(f // bn, f, bias=linear_bias),
			)
			block.tfc2 = nn.Sequential(
				nn.InstanceNorm2d(c, affine=norm_affine, eps=norm_eps), activation(), nn.Conv2d(c, c, 3, 1, 1, bias=conv_bias)
			)
			block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=conv_bias)

			self.blocks.append(block)
			in_c = c

	def forward(self, x: Tensor) -> Tensor:
		for block in self.blocks:
			s: Tensor = cast('Callable[[Tensor], Tensor]', block.shortcut)(x)
			x = cast('Callable[[Tensor], Tensor]', block.tfc1)(x)
			x = x + cast('Callable[[Tensor], Tensor]', block.tdf)(x)
			x = cast('Callable[[Tensor], Tensor]', block.tfc2)(x)
			x = x + s
		return x

class FreqPixelShuffle(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		scale: int,
		f: int,
		tfc_tdf_depth: int = 2,
		tfc_tdf_bn: int = 4,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()
		self.scale: int = scale
		self.conv: DSConv = DSConv(
			in_channels, out_channels * scale, activation=activation, norm_eps=norm_eps, norm_affine=norm_affine, bias=conv_bias
		)
		self.out_conv: TFC_TDF = TFC_TDF(
			out_channels,
			out_channels,
			tfc_tdf_depth,
			f,
			bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)

	def forward(self, x: Tensor) -> Tensor:
		x = self.conv(x)
		B, C_r, H, W = x.shape
		out_c: int = C_r // self.scale

		x = x.view(B, out_c, self.scale, H, W)

		x = x.permute(0, 1, 3, 4, 2).contiguous()
		x = x.view(B, out_c, H, W * self.scale)

		return self.out_conv(x)

class ProgressiveUpsampleHead(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		target_bins: int = 1025,
		in_bands: int = 62,
		upsample_scales: tuple[int, int, int, int] = (2, 2, 2, 2),
		tfc_tdf_depth: int = 2,
		tfc_tdf_bn: int = 4,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()
		self.target_bins: int = target_bins

		c: int = in_channels
		scale1, scale2, scale3, scale4 = upsample_scales
		f1: int = in_bands * scale1
		f2: int = f1 * scale2
		f3: int = f2 * scale3
		f4: int = f3 * scale4

		self.block1: FreqPixelShuffle = FreqPixelShuffle(
			c,
			c // 2,
			scale=scale1,
			f=f1,
			tfc_tdf_depth=tfc_tdf_depth,
			tfc_tdf_bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)
		self.block2: FreqPixelShuffle = FreqPixelShuffle(
			c // 2,
			c // 4,
			scale=scale2,
			f=f2,
			tfc_tdf_depth=tfc_tdf_depth,
			tfc_tdf_bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)
		self.block3: FreqPixelShuffle = FreqPixelShuffle(
			c // 4,
			c // 8,
			scale=scale3,
			f=f3,
			tfc_tdf_depth=tfc_tdf_depth,
			tfc_tdf_bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)
		self.block4: FreqPixelShuffle = FreqPixelShuffle(
			c // 8,
			c // 16,
			scale=scale4,
			f=f4,
			tfc_tdf_depth=tfc_tdf_depth,
			tfc_tdf_bn=tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)

		self.final_conv = nn.Conv2d(c // 16, out_channels, kernel_size=3, stride=1, padding='same', bias=conv_bias)

	def forward(self, x: Tensor) -> Tensor:

		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		if x.shape[-1] != self.target_bins:
			x = F.interpolate(x, size=(x.shape[2], self.target_bins), mode='bilinear', align_corners=False)

		return self.final_conv(x)

class SegmModel(nn.Module):
	def __init__(
		self,
		in_bands: int = 62,
		in_dim: int = 256,
		out_bins: int = 1025,
		out_channels: int = 4,
		base_channels: int = 64,
		base_depth: int = 2,
		num_hyperedges: int = 32,
		num_heads: int = 8,
		backbone_channels: tuple[int, int, int, int, int] | None = None,
		hyperace_k: int = 2,
		hyperace_l: int = 1,
		hyperace_c_h: float = 0.5,
		hyperace_c_l: float = 0.25,
		hyperace_c3ah_expansion: float = 1.0,
		hyperace_low_order_depth: int = 1,
		hyperace_low_order_kernel: int = 3,
		hyperace_low_order_expansion: float = 1.0,
		hyperace_out_channels: int | None = None,
		decoder_channels: list[int] | tuple[int, int, int, int] | None = None,
		decoder_block_depth: int = 1,
		decoder_block_kernel: int = 3,
		decoder_block_expansion: float = 0.5,
		upsample_scales: tuple[int, int, int, int] = (2, 2, 2, 2),
		upsample_tfc_tdf_depth: int = 2,
		upsample_tfc_tdf_bn: int = 4,
		activation: type[nn.Module] = nn.SiLU,
		norm_eps: float = 1e-8,
		*, norm_affine: bool = True,
		conv_bias: bool = False,
		linear_bias: bool = False,
	) -> None:
		super().__init__()

		self.backbone = Backbone(
			in_channels=in_dim,
			base_channels=base_channels,
			base_depth=base_depth,
			channels=backbone_channels,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
		)
		enc_channels: list[int] = self.backbone.out_channels
		c2, c3, c4, c5 = enc_channels

		hyperace_in_channels: list[int] = enc_channels
		hyperace_out_channels = c4 if hyperace_out_channels is None else hyperace_out_channels
		self.hyperace = HyperACE(
			hyperace_in_channels,
			hyperace_out_channels,
			num_hyperedges,
			num_heads,
			k=hyperace_k,
			l=hyperace_l,
			c_h=hyperace_c_h,
			c_l=hyperace_c_l,
			c3ah_expansion=hyperace_c3ah_expansion,
			low_order_depth=hyperace_low_order_depth,
			low_order_kernel=hyperace_low_order_kernel,
			low_order_expansion=hyperace_low_order_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)

		decoder_channels = [c2, c3, c4, c5] if decoder_channels is None else list(decoder_channels)
		self.decoder = Decoder(
			enc_channels,
			hyperace_out_channels,
			decoder_channels,
			block_depth=decoder_block_depth,
			block_kernel=decoder_block_kernel,
			block_expansion=decoder_block_expansion,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
		)

		self.upsample_head = ProgressiveUpsampleHead(
			in_channels=decoder_channels[0],
			out_channels=out_channels,
			target_bins=out_bins,
			in_bands=in_bands,
			upsample_scales=upsample_scales,
			tfc_tdf_depth=upsample_tfc_tdf_depth,
			tfc_tdf_bn=upsample_tfc_tdf_bn,
			activation=activation,
			norm_eps=norm_eps,
			norm_affine=norm_affine,
			conv_bias=conv_bias,
			linear_bias=linear_bias,
		)

	def forward(self, x: Tensor) -> Tensor:
		H, _W = x.shape[2:]

		enc_feats = self.backbone(x)

		h_ace_feats = self.hyperace(enc_feats)

		dec_feat = self.decoder(enc_feats, h_ace_feats)

		feat_time_restored: Tensor = F.interpolate(dec_feat, size=(H, dec_feat.shape[-1]), mode='bilinear', align_corners=False)

		return self.upsample_head(feat_time_restored)

