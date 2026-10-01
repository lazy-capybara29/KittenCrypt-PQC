"""End-to-end tests. Run from the project root with:  pytest"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from pqsteg import cli, fec_engine, pipeline, pqc_crypto, steg_core
from pqsteg.exceptions import (
    CapacityError,
    DecryptionError,
    ImageFormatError,
    IntegrityError,
    InvalidKeyError,
    PQStegError,
    QualityError,
)

SECRET = "Attack at dawn. — the quick brown fox".encode("utf-8")


def make_cover(size: int = 192, seed: int = 7) -> NDArray[np.uint8]:
    """A smooth, slightly noisy synthetic 'photo' (deterministic)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 100 + 50 * np.sin(x / 23.0) + 40 * np.cos(y / 31.0)
    channels = [base + shift + rng.normal(0, 3, (size, size)) for shift in (0, 20, -20)]
    image: NDArray[np.uint8] = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
    return image


def flip_hidden_bits(
    result: pipeline.EmbedResult, private_key: bytes, bit_indices: NDArray[np.intp]
) -> NDArray[np.uint8]:
    """Simulate channel noise: flip the pixel LSBs that carry the given payload bits."""
    seed = pqc_crypto.open_session(private_key, result.key_file).position_seed
    order = steg_core.pixel_order(result.stego.shape, 1, seed)
    noisy = result.stego.flatten()
    noisy[order[bit_indices]] ^= np.uint8(1)
    return noisy.reshape(result.stego.shape)


@pytest.fixture(scope="module")
def keypair() -> Tuple[bytes, bytes]:
    return pqc_crypto.generate_keypair()


@pytest.fixture(scope="module")
def other_keypair() -> Tuple[bytes, bytes]:
    return pqc_crypto.generate_keypair()


@pytest.fixture(scope="module")
def cover() -> NDArray[np.uint8]:
    return make_cover()


@pytest.fixture(scope="module")
def embedded(cover: NDArray[np.uint8], keypair: Tuple[bytes, bytes]) -> pipeline.EmbedResult:
    """One shared embedding reused by the read-only tests below."""
    return pipeline.embed(cover, SECRET, keypair[0])


# --------------------------------------------------------------------------- #
# Message integrity and image quality
# --------------------------------------------------------------------------- #
def test_round_trip_through_png_file(
    embedded: pipeline.EmbedResult, keypair: Tuple[bytes, bytes], tmp_path: Path
) -> None:
    path = tmp_path / "stego.png"
    steg_core.save_lossless_image(embedded.stego, path)
    reloaded = steg_core.load_rgb_image(path)
    assert np.array_equal(reloaded, embedded.stego)  # PNG is lossless
    assert pipeline.extract(reloaded, embedded.key_file, keypair[1]) == SECRET


def test_quality_and_minimal_change(cover: NDArray[np.uint8], embedded: pipeline.EmbedResult) -> None:
    assert embedded.report.psnr_db >= 42.0
    assert embedded.report.ssim >= 0.98
    assert sum(embedded.report.chi_square) > 0  # something did change
    assert np.abs(embedded.stego.astype(int) - cover.astype(int)).max() <= 1  # only LSBs


def test_cover_is_not_modified_in_place(keypair: Tuple[bytes, bytes]) -> None:
    cover = make_cover()
    original = cover.copy()
    pipeline.embed(cover, SECRET, keypair[0])
    assert np.array_equal(cover, original)


def test_binary_payload_with_two_bits_per_channel(
    cover: NDArray[np.uint8], keypair: Tuple[bytes, bytes]
) -> None:
    payload = np.random.default_rng(1).bytes(1500)
    result = pipeline.embed(cover, payload, keypair[0], bits_per_channel=2)
    assert result.report.psnr_db >= 42.0
    assert pipeline.extract(result.stego, result.key_file, keypair[1], bits_per_channel=2) == payload


def test_empty_secret_round_trips(cover: NDArray[np.uint8], keypair: Tuple[bytes, bytes]) -> None:
    result = pipeline.embed(cover, b"", keypair[0])
    assert pipeline.extract(result.stego, result.key_file, keypair[1]) == b""


def test_every_embedding_uses_fresh_randomness(
    cover: NDArray[np.uint8], keypair: Tuple[bytes, bytes]
) -> None:
    first = pipeline.embed(cover, SECRET, keypair[0])
    second = pipeline.embed(cover, SECRET, keypair[0])
    assert first.key_file != second.key_file
    assert not np.array_equal(first.stego, second.stego)  # different pixels chosen


def test_pixel_order_depends_on_seed(cover: NDArray[np.uint8]) -> None:
    def order(seed: bytes) -> NDArray[np.intp]:
        return steg_core.pixel_order(cover.shape, 1, seed)[:200]

    assert np.array_equal(order(b"a" * 32), order(b"a" * 32))  # deterministic
    assert not np.array_equal(order(b"a" * 32), order(b"b" * 32))
    with pytest.raises(InvalidKeyError):
        order(b"too short")


# --------------------------------------------------------------------------- #
# Robustness (forward error correction)
# --------------------------------------------------------------------------- #
def test_single_flipped_bit_is_repaired(
    embedded: pipeline.EmbedResult, keypair: Tuple[bytes, bytes]
) -> None:
    noisy = flip_hidden_bits(embedded, keypair[1], np.array([100], dtype=np.intp))
    assert not np.array_equal(noisy, embedded.stego)
    assert pipeline.extract(noisy, embedded.key_file, keypair[1]) == SECRET


def test_twenty_scattered_bit_flips_are_repaired(
    embedded: pipeline.EmbedResult, keypair: Tuple[bytes, bytes]
) -> None:
    # 20 flipped bits damage at most 20 bytes, below the 32 bytes/block RS(255,191) repairs.
    chosen = np.random.default_rng(3).choice(embedded.hidden_bytes * 8, size=20, replace=False)
    noisy = flip_hidden_bits(embedded, keypair[1], chosen.astype(np.intp))
    assert pipeline.extract(noisy, embedded.key_file, keypair[1]) == SECRET


def test_fec_repairs_up_to_capacity_and_detects_beyond() -> None:
    data = np.random.default_rng(5).bytes(1000)
    encoded = fec_engine.encode(data)

    def corrupt_blocks(errors_per_block: int) -> bytes:
        damaged = bytearray(encoded)
        for start in range(fec_engine.PREFIX_LEN, len(damaged), 255):
            end = min(start + 255, len(damaged))
            for position in range(start, min(start + errors_per_block, end)):
                damaged[position] ^= 0xFF
        return bytes(damaged)

    assert fec_engine.decode(corrupt_blocks(32)) == data  # exactly at the limit (12.5 %)
    with pytest.raises(IntegrityError):
        fec_engine.decode(corrupt_blocks(60))  # far beyond: refused, not silently wrong


def test_encoded_length_matches_encode() -> None:
    for size in (0, 1, 190, 191, 192, 1000):
        assert len(fec_engine.encode(bytes(size))) == fec_engine.encoded_length(size)


def test_heavy_noise_is_rejected_never_returns_wrong_data(
    embedded: pipeline.EmbedResult, keypair: Tuple[bytes, bytes]
) -> None:
    n_bits = embedded.hidden_bytes * 8
    chosen = np.random.default_rng(4).choice(n_bits, size=n_bits // 3, replace=False)
    noisy = flip_hidden_bits(embedded, keypair[1], chosen.astype(np.intp))
    with pytest.raises(PQStegError):
        pipeline.extract(noisy, embedded.key_file, keypair[1])


# --------------------------------------------------------------------------- #
# Key rejection
# --------------------------------------------------------------------------- #
def test_wrong_private_key_is_rejected(
    embedded: pipeline.EmbedResult, other_keypair: Tuple[bytes, bytes]
) -> None:
    # A wrong private key yields a wrong pixel order: the receiver reads noise.
    with pytest.raises(IntegrityError):
        pipeline.extract(embedded.stego, embedded.key_file, other_keypair[1])


def test_key_file_from_another_message_is_rejected(
    cover: NDArray[np.uint8], embedded: pipeline.EmbedResult, keypair: Tuple[bytes, bytes]
) -> None:
    other = pipeline.embed(cover, b"a different message", keypair[0])
    with pytest.raises(IntegrityError):
        pipeline.extract(embedded.stego, other.key_file, keypair[1])


def test_malformed_keys_are_rejected(
    cover: NDArray[np.uint8], embedded: pipeline.EmbedResult, keypair: Tuple[bytes, bytes]
) -> None:
    with pytest.raises(InvalidKeyError):
        pipeline.embed(cover, SECRET, b"too short")
    with pytest.raises(InvalidKeyError):
        pipeline.extract(embedded.stego, embedded.key_file[:-5], keypair[1])
    with pytest.raises(InvalidKeyError):
        pipeline.extract(embedded.stego, embedded.key_file, b"\x00" * pqc_crypto.PRIVATE_KEY_SIZE)


def test_wrong_bit_depth_is_rejected(
    embedded: pipeline.EmbedResult, keypair: Tuple[bytes, bytes]
) -> None:
    with pytest.raises(IntegrityError):
        pipeline.extract(embedded.stego, embedded.key_file, keypair[1], bits_per_channel=2)


def test_tampered_ciphertext_fails_authentication(keypair: Tuple[bytes, bytes]) -> None:
    session = pqc_crypto.new_session(keypair[0])
    blob = pqc_crypto.encrypt(session, b"hello")
    assert pqc_crypto.decrypt(session, blob) == b"hello"
    with pytest.raises(DecryptionError):
        pqc_crypto.decrypt(session, blob[:-1] + bytes([blob[-1] ^ 0x01]))


# --------------------------------------------------------------------------- #
# Capacity, fill limits, quality gate, file format
# --------------------------------------------------------------------------- #
def test_capacity_formula_and_error(keypair: Tuple[bytes, bytes]) -> None:
    assert steg_core.capacity_bytes((10, 10, 3), 1) == 10 * 10 * 3 // 8
    assert steg_core.capacity_bytes((10, 10, 3), 2) == 10 * 10 * 3 * 2 // 8
    with pytest.raises(CapacityError):
        pipeline.embed(make_cover(size=16), SECRET, keypair[0])  # holds 96 bytes only


def test_enhanced_privacy_limit_is_exact(keypair: Tuple[bytes, bytes]) -> None:
    big_cover = make_cover(size=384)
    limit = pipeline.max_secret_bytes(big_cover.shape, 1, pipeline.ENHANCED_MAX_FILL)
    assert limit > len(SECRET)  # this cover can carry a normal short message

    fits = pipeline.embed(
        big_cover, bytes(limit), keypair[0], max_fill=pipeline.ENHANCED_MAX_FILL
    )
    assert fits.fill <= pipeline.ENHANCED_MAX_FILL
    with pytest.raises(CapacityError):  # one byte more is refused
        pipeline.embed(
            big_cover, bytes(limit + 1), keypair[0], max_fill=pipeline.ENHANCED_MAX_FILL
        )
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            pipeline.embed(big_cover, SECRET, keypair[0], max_fill=bad)


def test_quality_gate_can_reject(cover: NDArray[np.uint8], keypair: Tuple[bytes, bytes]) -> None:
    with pytest.raises(QualityError):
        pipeline.embed(cover, SECRET, keypair[0], min_psnr=99.0)


def test_lossy_output_format_is_refused(cover: NDArray[np.uint8], tmp_path: Path) -> None:
    with pytest.raises(ImageFormatError):
        steg_core.save_lossless_image(cover, tmp_path / "stego.jpg")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class _FakeTerminal(io.StringIO):
    """Stands in for an interactive terminal whose user types the given text."""

    def isatty(self) -> bool:
        return True


def _prepare(tmp_path: Path, size: int) -> Tuple[str, str]:
    cover_path = tmp_path / "cover.png"
    steg_core.save_lossless_image(make_cover(size=size), cover_path)
    keys = str(tmp_path / "bob")
    assert cli.main(["generate-keys", "--out-prefix", keys]) == 0
    return str(cover_path), keys


def test_cli_round_trip_and_key_rejection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cover_path, keys = _prepare(tmp_path, 192)
    stego = str(tmp_path / "stego.png")
    secret_path = tmp_path / "secret.bin"
    secret_path.write_bytes(SECRET)

    assert cli.main(["generate-keys", "--out-prefix", keys]) == 1  # never overwrites keys
    assert cli.main([
        "embed", "--cover", cover_path, "--secret-file", str(secret_path),
        "--public-key", keys + ".pub", "--out", stego, "--privacy", "standard",
    ]) == 0
    assert "PSNR" in capsys.readouterr().out
    assert (tmp_path / "stego.kex").exists()  # the second file

    recovered = tmp_path / "recovered.bin"
    assert cli.main(["extract", "--stego", stego, "--private-key", keys + ".key",
                     "--out", str(recovered)]) == 0
    assert recovered.read_bytes() == SECRET

    other = str(tmp_path / "eve")
    assert cli.main(["generate-keys", "--out-prefix", other]) == 0
    assert cli.main(["extract", "--stego", stego, "--private-key", other + ".key",
                     "--out", str(tmp_path / "nope.bin")]) == 1


def test_cli_enhanced_privacy_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    small_cover, keys = _prepare(tmp_path, 192)  # enhanced limit here is only ~26 bytes
    common = ["--public-key", keys + ".pub", "--out", str(tmp_path / "s.png"), "--privacy", "enhanced"]

    assert cli.main(["embed", "--cover", small_cover, "--message", SECRET.decode(), *common]) == 1
    assert "enhanced privacy allows at most" in capsys.readouterr().err

    assert cli.main(["embed", "--cover", small_cover, "--message", "tiny", *common]) == 0
    assert "enhanced privacy" in capsys.readouterr().out


def test_cli_asks_in_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cover_path, keys = _prepare(tmp_path, 192)
    args = ["embed", "--cover", cover_path, "--message", "tiny",
            "--public-key", keys + ".pub", "--out", str(tmp_path / "s.png")]

    monkeypatch.setattr(sys, "stdin", _FakeTerminal("y\n"))
    assert cli.main(args) == 0
    output = capsys.readouterr().out
    assert "Enhanced privacy mode hides at most" in output and "mode            : enhanced" in output

    monkeypatch.setattr(sys, "stdin", _FakeTerminal("n\n"))
    assert cli.main(args) == 0
    assert "mode            : standard" in capsys.readouterr().out
