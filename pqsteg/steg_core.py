"""Module C - keyed pseudo-random LSB embedding.

How it works, in four steps:

1. An image with H x W pixels, 3 colour channels and k bits per channel has
   N = H * W * 3 * k "slots" (one slot = one least-significant-bit position of one
   channel value). Slot s means: channel value number s // k, bit number s % k.
2. A secret 32-byte position seed (derived from the ML-KEM secret, see pqc_crypto) feeds
   SHAKE-256, a cryptographic extendable-output function, used as a deterministic CSPRNG.
   (Python's `secrets` cannot be seeded, so it cannot reproduce the order for the receiver.)
3. We draw one random 64-bit sort key per slot and sort the slots by it. That is a
   uniformly random permutation of all slots, reproducible only with the seed.
   Payload bit i goes into slot order[i].
4. Slots that are not needed stay untouched. How many are used (the "fill") matters most
   for detectability; see pipeline.py.

Memory: the order costs roughly 23 bytes per slot (measured: ~290 MB and ~2 s for a
3-megapixel image). Crop or downscale very large photos.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .exceptions import ImageFormatError, InvalidKeyError

LOSSLESS_SUFFIXES = (".png", ".bmp")
POSITION_SEED_SIZE = 32

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Image files
# --------------------------------------------------------------------------- #
def load_rgb_image(path: PathLike) -> NDArray[np.uint8]:
    """Load any Pillow-readable image as an (H, W, 3) uint8 array. Alpha is dropped."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def ensure_lossless_path(path: PathLike) -> None:
    """Refuse output formats that would destroy the hidden bits (e.g. JPEG)."""
    if Path(path).suffix.lower() not in LOSSLESS_SUFFIXES:
        raise ImageFormatError(
            f"output must be a lossless format {LOSSLESS_SUFFIXES}; "
            "lossy formats such as JPEG destroy the hidden data"
        )


def save_lossless_image(image: NDArray[np.uint8], path: PathLike) -> None:
    ensure_lossless_path(path)
    Image.fromarray(image).save(path)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def _slot_count(shape: Tuple[int, ...], bits_per_channel: int) -> int:
    if len(shape) != 3 or shape[2] != 3:
        raise ImageFormatError(f"expected an (H, W, 3) image, got shape {shape}")
    if bits_per_channel not in (1, 2):
        raise ValueError("bits_per_channel must be 1 or 2")
    return shape[0] * shape[1] * shape[2] * bits_per_channel


def capacity_bytes(shape: Tuple[int, ...], bits_per_channel: int = 1) -> int:
    """C_max = W * H * 3 * k / 8, rounded down to whole bytes."""
    return _slot_count(shape, bits_per_channel) // 8


def pixel_order(
    shape: Tuple[int, ...], bits_per_channel: int, position_seed: bytes
) -> NDArray[np.intp]:
    """The secret slot order. Same seed + image size + bit depth = same order."""
    if len(position_seed) != POSITION_SEED_SIZE:
        raise InvalidKeyError(f"position seed must be {POSITION_SEED_SIZE} bytes")
    n_slots = _slot_count(shape, bits_per_channel)
    # Image size and bit depth are mixed in, so a wrong --bits gives a completely wrong order.
    material = (
        b"pqsteg|order|" + position_seed + struct.pack(">IIB", shape[0], shape[1], bits_per_channel)
    )
    keystream = hashlib.shake_256(material).digest(8 * n_slots)
    return np.argsort(np.frombuffer(keystream, dtype=">u8"), kind="stable")


def embed_bytes(
    cover: NDArray[np.uint8], data: bytes, order: NDArray[np.intp], bits_per_channel: int
) -> NDArray[np.uint8]:
    """Return a copy of `cover` with `data` hidden in it. The caller checks capacity."""
    k = bits_per_channel
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    slots = order[: bits.size]
    channel_index, bit_plane = slots // k, slots % k

    flat = cover.flatten()  # always a copy
    # One pass per bit plane: within a plane every slot is a different channel value,
    # so the vectorised assignment cannot collide with itself.
    for plane in range(k):
        chosen = bit_plane == plane
        idx = channel_index[chosen]
        keep = np.uint8(0xFF ^ (1 << plane))
        flat[idx] = (flat[idx] & keep) | (bits[chosen] << np.uint8(plane))
    return flat.reshape(cover.shape)


def extract_bytes(
    stego: NDArray[np.uint8], order: NDArray[np.intp], bits_per_channel: int, n_bytes: int
) -> bytes:
    """Read the first `n_bytes` hidden bytes."""
    k = bits_per_channel
    slots = order[: n_bytes * 8]
    bits = (stego.reshape(-1)[slots // k] >> (slots % k).astype(np.uint8)) & np.uint8(1)
    return np.packbits(bits).tobytes()
