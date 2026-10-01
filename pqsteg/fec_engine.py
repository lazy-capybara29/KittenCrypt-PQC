"""Module B - forward error correction with Reed-Solomon over GF(2^8).

Layout produced by encode():

    [ length prefix: 20 bytes ][ body ]

* body   : the data is cut into 191-byte chunks and every chunk gets 64 parity bytes,
           i.e. RS(255, 191) codewords. Each codeword repairs up to 32 wrong BYTES
           (12.5 %). Cost: 64/191 = 33.5 % extra bytes.
* prefix : the data length (4 bytes) protected by its own RS code with 16 parity bytes
           (repairs 8 wrong bytes out of 20). The extractor must know the length before
           it can read the body, so it gets stronger protection.

One flipped bit ruins one byte, so a random bit-error rate of p damages about 8p of the
bytes: RS(255,191) tolerates roughly a 1.5 % random bit-error rate. Beyond that, decode()
raises IntegrityError (or, very rarely, "repairs" to wrong data, which the AES-GCM tag
then catches).

The parameters are constants, not stored in the stream: sender and receiver must run the
same version of this file.
"""

from __future__ import annotations

import struct

from reedsolo import ReedSolomonError, RSCodec

from .exceptions import IntegrityError

BODY_PARITY = 64
CHUNK = 255 - BODY_PARITY  # data bytes per Reed-Solomon block
PREFIX_PARITY = 16
PREFIX_LEN = 4 + PREFIX_PARITY

_body_codec = RSCodec(BODY_PARITY)
_prefix_codec = RSCodec(PREFIX_PARITY)


def encoded_length(data_length: int) -> int:
    """Size of encode(data) for data of the given length."""
    chunks = -(-data_length // CHUNK)  # ceiling division
    return PREFIX_LEN + data_length + chunks * BODY_PARITY


def encode(data: bytes) -> bytes:
    """Return length prefix + Reed-Solomon encoded body."""
    prefix = bytes(_prefix_codec.encode(struct.pack(">I", len(data))))
    return prefix + bytes(_body_codec.encode(data))


def total_length(prefix: bytes) -> int:
    """Repair the length prefix; return the size of the whole encoded stream it announces."""
    if len(prefix) != PREFIX_LEN:
        raise IntegrityError(f"length prefix must be {PREFIX_LEN} bytes, got {len(prefix)}")
    try:
        (data_length,) = struct.unpack(">I", bytes(_prefix_codec.decode(prefix)[0]))
    except (ReedSolomonError, ValueError, struct.error) as exc:
        raise IntegrityError("length prefix is unrecoverable") from exc
    return encoded_length(int(data_length))


def decode(data: bytes) -> bytes:
    """Repair and return the original data, or raise IntegrityError."""
    expected = total_length(data[:PREFIX_LEN])
    if len(data) != expected:
        raise IntegrityError(f"stream has {len(data)} bytes but the prefix announces {expected}")
    try:
        return bytes(_body_codec.decode(data[PREFIX_LEN:])[0])
    except (ReedSolomonError, ValueError) as exc:
        raise IntegrityError("too many errors for Reed-Solomon to repair") from exc
