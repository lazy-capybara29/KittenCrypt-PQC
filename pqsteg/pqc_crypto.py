"""Module A - hybrid post-quantum encryption (ML-KEM-768 + AES-256-GCM).

One ML-KEM encapsulation per message gives a fresh 32-byte shared secret K. From K we
derive two independent secrets:

    aes_key        encrypts the payload with AES-256-GCM
    position_seed  decides WHICH pixel bits carry the payload (used by steg_core)

Only the holder of the receiver's private key can recover K, so only they can decrypt
the message or even find where it is hidden.

The sender delivers two things:

  * the KEY FILE: the 1088-byte ML-KEM ciphertext ("sealed envelope"). Public.
  * the PAYLOAD, hidden in the image:  nonce (12) | GCM tag (16) | encrypted secret

The key file is also fed to GCM as "associated data", so a payload only decrypts
together with the key file it was made for. No lengths are stored: the file sizes and
the error-correction layer already fix them.

Wrong private key: ML-KEM never raises here. It silently returns a different K, hence a
different pixel order, so the receiver reads noise and fails earlier, in the image
reading step (IntegrityError). DecryptionError means the bits were read fine but the
GCM tag does not verify, i.e. tampering.

To switch to liboqs, change only the ML_KEM_768 calls in generate_keypair, new_session
and open_session.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import NamedTuple, Tuple

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from kyber_py.ml_kem import ML_KEM_768

from .exceptions import DecryptionError, IntegrityError, InvalidKeyError

PUBLIC_KEY_SIZE = 1184
PRIVATE_KEY_SIZE = 2400
KEY_FILE_SIZE = 1088
NONCE_SIZE = 12
TAG_SIZE = 16
OVERHEAD = NONCE_SIZE + TAG_SIZE  # bytes the payload adds to the secret


class Session(NamedTuple):
    """Everything derived from one ML-KEM encapsulation."""

    key_file: bytes  # public: sent next to the image
    aes_key: bytes  # secret
    position_seed: bytes  # secret


def generate_keypair() -> Tuple[bytes, bytes]:
    """Return (public_key, private_key)."""
    public_key, private_key = ML_KEM_768.keygen()
    return bytes(public_key), bytes(private_key)


def new_session(public_key: bytes) -> Session:
    """Sender side: fresh K plus the key file the receiver will need."""
    _require_size(public_key, PUBLIC_KEY_SIZE, "public key")
    try:
        shared_secret, key_file = ML_KEM_768.encaps(public_key)
    except ValueError as exc:
        raise InvalidKeyError(f"public key rejected: {exc}") from exc
    return _derive(bytes(shared_secret), bytes(key_file))


def open_session(private_key: bytes, key_file: bytes) -> Session:
    """Receiver side: recover K from the key file and derive the same secrets."""
    _require_size(private_key, PRIVATE_KEY_SIZE, "private key")
    _require_size(key_file, KEY_FILE_SIZE, "key file")
    try:
        shared_secret = ML_KEM_768.decaps(private_key, key_file)
    except ValueError as exc:
        raise InvalidKeyError(f"private key rejected: {exc}") from exc
    return _derive(bytes(shared_secret), key_file)


def encrypt(session: Session, plaintext: bytes) -> bytes:
    """Return nonce | tag | encrypted plaintext."""
    nonce = get_random_bytes(NONCE_SIZE)
    cipher = AES.new(session.aes_key, AES.MODE_GCM, nonce=nonce)
    cipher.update(session.key_file)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce + tag + ciphertext


def decrypt(session: Session, blob: bytes) -> bytes:
    """Verify the tag and return the plaintext."""
    if len(blob) < OVERHEAD:
        raise IntegrityError("encrypted payload is truncated")
    nonce, tag, ciphertext = blob[:NONCE_SIZE], blob[NONCE_SIZE:OVERHEAD], blob[OVERHEAD:]
    cipher = AES.new(session.aes_key, AES.MODE_GCM, nonce=nonce)
    cipher.update(session.key_file)
    try:
        return cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError as exc:
        raise DecryptionError("authentication failed: the hidden data was modified") from exc


def _derive(shared_secret: bytes, key_file: bytes) -> Session:
    def prf(label: bytes) -> bytes:  # HMAC-SHA256 keyed with K, one label per secret
        return hmac.new(shared_secret, label, hashlib.sha256).digest()

    return Session(key_file, prf(b"pqsteg aes-256-gcm key"), prf(b"pqsteg pixel order seed"))


def _require_size(value: bytes, expected: int, what: str) -> None:
    if len(value) != expected:
        raise InvalidKeyError(f"{what} must be {expected} bytes, got {len(value)}")
