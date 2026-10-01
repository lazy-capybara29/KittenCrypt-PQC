"""Module D - how much did embedding change the image?

Three numbers, all comparing the cover image I with the stego image I':

* PSNR       10 * log10(255^2 / MSE) in dB (MSE = mean squared pixel difference).
             Higher is better; identical images give infinity.
* SSIM       structural similarity (scikit-image), 1.0 = identical.
* chi-square distance between the histograms h (cover) and h' (stego) of each colour
             channel:  sum over bins of (h - h')^2 / (h + h'), skipping empty bins.
             0 = identical; it grows with the number of modified pixels. It has no
             universal pass/fail threshold, so it is reported, not enforced.

These measure DISTORTION. They do not prove that a dedicated steganalysis tool cannot
find the message (see README).
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity

PSNR_MIN = 42.0
SSIM_MIN = 0.98


class QualityReport(NamedTuple):
    psnr_db: float
    ssim: float
    chi_square: Tuple[float, float, float]  # per channel: R, G, B


def psnr(cover: NDArray[np.uint8], stego: NDArray[np.uint8]) -> float:
    difference = cover.astype(np.int32) - stego.astype(np.int32)  # int32: uint8 would wrap
    mse = float(np.mean(difference.astype(np.float64) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(255.0**2 / mse))


def ssim(cover: NDArray[np.uint8], stego: NDArray[np.uint8]) -> float:
    # scikit-image ships no type hints, hence the ignore.
    score = structural_similarity(  # type: ignore[no-untyped-call]
        cover, stego, channel_axis=2, data_range=255
    )
    return float(score)


def chi_square(cover: NDArray[np.uint8], stego: NDArray[np.uint8]) -> Tuple[float, float, float]:
    distances = []
    for channel in range(3):
        before = np.bincount(cover[..., channel].ravel(), minlength=256)
        after = np.bincount(stego[..., channel].ravel(), minlength=256)
        total = (before + after).astype(np.float64)
        used = total > 0
        gap = (before - after).astype(np.float64) ** 2
        distances.append(float(np.sum(gap[used] / total[used])))
    return distances[0], distances[1], distances[2]


def measure(cover: NDArray[np.uint8], stego: NDArray[np.uint8]) -> QualityReport:
    return QualityReport(psnr(cover, stego), ssim(cover, stego), chi_square(cover, stego))
