"""The pipeline: wires modules A-D together in a fixed order.

    embed:    secret -> encrypt -> Reed-Solomon -> random-position LSB -> stego image
                  \\-> key file (a separate file, sent next to the image)
    extract:  key file + private key -> pixel order;  stego image -> read bits ->
              Reed-Solomon repair -> decrypt -> secret

Fill and privacy modes
----------------------
"Fill" is the fraction of the image's LSB slots that carry payload. The more you use, the
easier the change is to detect. RS analysis (a standard steganalysis test) run on real
photos read the true fill almost exactly at 5-30 % and was near its noise floor at 1 %:

    STANDARD_MAX_FILL  70 %  allows big secrets; easy to detect statistically
    ENHANCED_MAX_FILL   1 %  small secrets only; hard for that test to tell from noise

Neither is a guarantee: stronger (machine-learning) detectors may still find LSB changes.

Errors (what each means for the receiver):
    InvalidKeyError  malformed key or key file
    IntegrityError   hidden bits not found or not repairable: wrong private key or key
                     file, wrong --bits, or the image was re-encoded
    DecryptionError  bits recovered fine but the GCM tag failed (tampering)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import fec_engine, metrics, pqc_crypto, steg_core
from .exceptions import CapacityError, IntegrityError, QualityError

STANDARD_MAX_FILL = 0.7
ENHANCED_MAX_FILL = 0.01


class EmbedResult(NamedTuple):
    stego: NDArray[np.uint8]
    key_file: bytes  # send this next to the image
    report: metrics.QualityReport
    secret_bytes: int
    hidden_bytes: int  # bytes actually written into the image (encrypted + error correction)
    capacity_bytes: int
    fill: float  # hidden_bytes / capacity_bytes


def _hidden_size(secret_size: int) -> int:
    """Bytes written into the image for a secret of this size."""
    return fec_engine.encoded_length(pqc_crypto.OVERHEAD + secret_size)


def max_secret_bytes(shape: tuple[int, ...], bits_per_channel: int, max_fill: float) -> int:
    """Largest secret that stays within `max_fill` of this image's capacity."""
    budget = int(max_fill * steg_core.capacity_bytes(shape, bits_per_channel))
    if _hidden_size(0) > budget:
        return 0
    low, high = 0, budget  # binary search: _hidden_size grows with the secret size
    while low < high:
        middle = (low + high + 1) // 2
        if _hidden_size(middle) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def embed(
    cover: NDArray[np.uint8],
    secret: bytes,
    public_key: bytes,
    *,
    bits_per_channel: int = 1,
    max_fill: float = STANDARD_MAX_FILL,
    min_psnr: float = metrics.PSNR_MIN,
    min_ssim: float = metrics.SSIM_MIN,
) -> EmbedResult:
    if not 0.0 < max_fill <= 1.0:
        raise ValueError("max_fill must be greater than 0 and at most 1")

    session = pqc_crypto.new_session(public_key)  # A
    framed = fec_engine.encode(pqc_crypto.encrypt(session, secret))  # B

    capacity = steg_core.capacity_bytes(cover.shape, bits_per_channel)
    budget = int(max_fill * capacity)
    if len(framed) > budget:
        raise CapacityError(
            f"the secret needs {len(framed)} hidden bytes but at most {budget} are allowed "
            f"({max_fill:.2%} of this image's {capacity}-byte capacity); "
            "use a larger image or a smaller secret"
        )

    order = steg_core.pixel_order(cover.shape, bits_per_channel, session.position_seed)  # C
    stego = steg_core.embed_bytes(cover, framed, order, bits_per_channel)

    report = metrics.measure(cover, stego)  # D
    if report.psnr_db < min_psnr or report.ssim < min_ssim:
        raise QualityError(
            f"quality gate failed: PSNR {report.psnr_db:.2f} dB (need >= {min_psnr}), "
            f"SSIM {report.ssim:.4f} (need >= {min_ssim}); "
            "try --bits 1 or a larger, more detailed cover image"
        )
    return EmbedResult(
        stego, session.key_file, report, len(secret), len(framed), capacity, len(framed) / capacity
    )


def extract(
    stego: NDArray[np.uint8],
    key_file: bytes,
    private_key: bytes,
    *,
    bits_per_channel: int = 1,
) -> bytes:
    session = pqc_crypto.open_session(private_key, key_file)
    order = steg_core.pixel_order(stego.shape, bits_per_channel, session.position_seed)

    # The length is unknown, so read the small length prefix first, then the rest.
    prefix = steg_core.extract_bytes(stego, order, bits_per_channel, fec_engine.PREFIX_LEN)
    total = fec_engine.total_length(prefix)
    if total > steg_core.capacity_bytes(stego.shape, bits_per_channel):
        raise IntegrityError("hidden length exceeds image capacity: wrong key, key file or --bits?")
    framed = steg_core.extract_bytes(stego, order, bits_per_channel, total)

    return pqc_crypto.decrypt(session, fec_engine.decode(framed))
