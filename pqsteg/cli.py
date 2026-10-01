"""Command line interface: generate-keys, embed, extract.

    pqsteg generate-keys --out-prefix alice
    pqsteg embed --cover cover.png --message "meet at 6" --public-key alice.pub --out stego.png
    pqsteg extract --stego stego.png --private-key alice.key

`embed` writes two files, stego.png and stego.kex. Send both. `extract` looks for the
.kex file next to the image (same name).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, pipeline
from .exceptions import PQStegError
from .pqc_crypto import generate_keypair
from .steg_core import ensure_lossless_path, load_rgb_image, save_lossless_image


def _key_file_path(image_path: str) -> Path:
    """stego.png -> stego.kex"""
    return Path(image_path).with_suffix(".kex")


def _write_new_file(path: Path, data: bytes, mode: int) -> None:
    """Create a file, refusing to overwrite an existing one (keys are not replaceable)."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _wants_enhanced_privacy(choice: Optional[str], secret_size: int, limit: int) -> bool:
    """--privacy decides; otherwise ask when running in a terminal, else standard."""
    if choice is not None:
        return choice == "enhanced"
    if not sys.stdin.isatty():
        return False
    print(f"Enhanced privacy mode hides at most {limit} bytes of secret in this image "
          f"(yours is {secret_size} bytes).")
    print(f"It uses only {pipeline.ENHANCED_MAX_FILL:.0%} of the image, which is much harder "
          "to detect than standard mode.")
    try:
        answer = input("Use enhanced privacy? [y/N] ")
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def cmd_generate_keys(args: argparse.Namespace) -> int:
    public_key, private_key = generate_keypair()
    public_path, private_path = Path(f"{args.out_prefix}.pub"), Path(f"{args.out_prefix}.key")
    _write_new_file(public_path, public_key, 0o644)
    _write_new_file(private_path, private_key, 0o600)
    print(f"public key  -> {public_path}  (share this)")
    print(f"private key -> {private_path}  (keep secret)")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    ensure_lossless_path(args.out)  # fail early, before any heavy work
    secret = (
        args.message.encode("utf-8") if args.message is not None
        else Path(args.secret_file).read_bytes()
    )
    public_key = Path(args.public_key).read_bytes()
    cover = load_rgb_image(args.cover)

    limit = pipeline.max_secret_bytes(cover.shape, args.bits, pipeline.ENHANCED_MAX_FILL)
    enhanced = _wants_enhanced_privacy(args.privacy, len(secret), limit)
    max_fill = pipeline.ENHANCED_MAX_FILL if enhanced else pipeline.STANDARD_MAX_FILL
    if enhanced and len(secret) > limit:
        print(f"error: enhanced privacy allows at most {limit} bytes for this image but the "
              f"secret is {len(secret)} bytes; use a larger image, a smaller secret, "
              "or standard mode", file=sys.stderr)
        return 1

    result = pipeline.embed(
        cover, secret, public_key, bits_per_channel=args.bits, max_fill=max_fill
    )
    key_path = _key_file_path(args.out)
    save_lossless_image(result.stego, args.out)
    key_path.write_bytes(result.key_file)

    r, g, b = result.report.chi_square
    print(f"mode            : {'enhanced privacy' if enhanced else 'standard'} "
          f"(up to {max_fill:.0%} of the image)")
    print(f"stego image     : {args.out}")
    print(f"key file        : {key_path}  (send it together with the image)")
    print(f"secret          : {result.secret_bytes} bytes")
    print(f"hidden in image : {result.hidden_bytes} bytes "
          f"({result.fill:.2%} of {result.capacity_bytes} bytes capacity)")
    print(f"PSNR            : {result.report.psnr_db:.2f} dB  (>= 42 required)")
    print(f"SSIM            : {result.report.ssim:.5f}  (>= 0.98 required)")
    print(f"chi-square dist : R {r:.1f}  G {g:.1f}  B {b:.1f}  (0 = identical)")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    private_key = Path(args.private_key).read_bytes()
    key_file = _key_file_path(args.stego).read_bytes()
    stego = load_rgb_image(args.stego)

    secret = pipeline.extract(stego, key_file, private_key, bits_per_channel=args.bits)

    if args.out is not None:
        Path(args.out).write_bytes(secret)
        print(f"authenticated and decrypted {len(secret)} bytes -> {args.out}")
        return 0
    try:
        print(secret.decode("utf-8"))
    except UnicodeDecodeError:
        print("recovered data is binary, not text; rerun with --out FILE", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pqsteg", description="Post-quantum steganography (ML-KEM-768 + AES-256-GCM)."
    )
    parser.add_argument("--version", action="version", version=f"pqsteg {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    keys = commands.add_parser("generate-keys", help="create an ML-KEM-768 key pair")
    keys.add_argument("--out-prefix", default="receiver", help="writes PREFIX.pub and PREFIX.key")
    keys.set_defaults(handler=cmd_generate_keys)

    embed = commands.add_parser("embed", help="hide a secret in a cover image")
    embed.add_argument("--cover", required=True, help="cover image (any format Pillow reads)")
    source = embed.add_mutually_exclusive_group(required=True)
    source.add_argument("--message", help="secret text")
    source.add_argument("--secret-file", help="file to hide")
    embed.add_argument("--public-key", required=True, help="receiver's .pub file")
    embed.add_argument("--out", required=True, help="stego image, must end in .png or .bmp")
    embed.add_argument("--bits", type=int, choices=(1, 2), default=1, help="LSBs per channel")
    embed.add_argument("--privacy", choices=("standard", "enhanced"),
                       help="skip the question; default: ask in a terminal, else standard")
    embed.set_defaults(handler=cmd_embed)

    extract = commands.add_parser("extract", help="recover a secret from a stego image")
    extract.add_argument("--stego", required=True, help="stego image (its .kex file must be next to it)")
    extract.add_argument("--private-key", required=True, help="your .key file")
    extract.add_argument("--bits", type=int, choices=(1, 2), default=1, help="must match embed")
    extract.add_argument("--out", help="write the secret to this file (default: print as text)")
    extract.set_defaults(handler=cmd_extract)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (PQStegError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

