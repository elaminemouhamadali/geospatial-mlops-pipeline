from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def build_grid(H: int, W: int, T: int, S: int) -> Tuple[List[int], List[int]]:
    ys = list(range(0, max(1, H - T + 1), S))
    if ys[-1] != H - T:
        ys.append(max(0, H - T))
    xs = list(range(0, max(1, W - T + 1), S))
    if xs[-1] != W - T:
        xs.append(max(0, W - T))
    return ys, xs

def _to_channels(x: torch.Tensor, out_channels: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [C,H,W], got shape {tuple(x.shape)}")

        c = int(x.shape[0])

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if c == out_channels:
            return x

        if c == 1 and out_channels > 1:
            return x.repeat(out_channels, 1, 1)

        if c > out_channels:
            return x[:out_channels]

        pad = out_channels - c
        return torch.cat([x, x[-1:].repeat(pad, 1, 1)], dim=0)


def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize postprocessor output to [B,C,H,W].

    Accepts:
      [B,H,W]   -> [B,1,H,W]
      [B,1,H,W] -> unchanged
      [B,C,H,W] -> unchanged
    """
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x).__name__}")

    if x.ndim == 3:
        return x.unsqueeze(1)

    if x.ndim == 4:
        return x

    raise ValueError(f"Expected [B,H,W] or [B,C,H,W], got {tuple(x.shape)}")


def _squeeze_single_channel(x: np.ndarray) -> np.ndarray:
    """
    Convert [1,H,W] -> [H,W], leave [C,H,W] unchanged.
    """
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    return x


def _extract_context_window(
    *,
    context: torch.Tensor,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    image_height: int,
    image_width: int,
    tile_size: int,
) -> torch.Tensor:
    if context.ndim != 3:
        raise ValueError(f"context must be [C,H,W], got {tuple(context.shape)}")

    _, context_height, context_width = context.shape

    if context_height == image_height and context_width == image_width:
        ctx = context[:, y0:y1, x0:x1]
        return _pad_to_size(ctx, tile_size, tile_size)

    cy0 = int(round(y0 * context_height / image_height))
    cy1 = int(round(y1 * context_height / image_height))
    cx0 = int(round(x0 * context_width / image_width))
    cx1 = int(round(x1 * context_width / image_width))

    cy0 = max(0, min(cy0, context_height - 1))
    cy1 = max(cy0 + 1, min(cy1, context_height))
    cx0 = max(0, min(cx0, context_width - 1))
    cx1 = max(cx0 + 1, min(cx1, context_width))

    ctx = context[:, cy0:cy1, cx0:cx1].unsqueeze(0)
    ctx = F.interpolate(
        ctx,
        size=(tile_size, tile_size),
        mode="bilinear",
        align_corners=False,
    )
    return ctx.squeeze(0)


def _pad_to_size(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")

    _, h, w = x.shape

    pad_h = max(0, height - h)
    pad_w = max(0, width - w)

    if pad_h == 0 and pad_w == 0:
        return x

    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
