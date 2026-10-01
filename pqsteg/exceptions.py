"""Exceptions. Everything this project raises on purpose derives from PQStegError."""


class PQStegError(Exception):
    """Base class for all errors raised by this project."""


class InvalidKeyError(PQStegError):
    """A key or key file is malformed (wrong size or unusable)."""


class DecryptionError(PQStegError):
    """AES-256-GCM authentication failed: the hidden data was modified."""


class IntegrityError(PQStegError):
    """The hidden bits could not be found or repaired (wrong key, re-encoded image, ...)."""


class CapacityError(PQStegError):
    """The payload does not fit: larger than the image, or above the allowed fill."""


class QualityError(PQStegError):
    """The stego image failed the PSNR / SSIM quality gate."""


class ImageFormatError(PQStegError):
    """The image is unusable: wrong pixel layout, or a lossy output format."""
