# pqsteg

Hide a secret message or file inside a PNG/BMP image, encrypted with **ML-KEM-768 (Kyber) + AES-256-GCM**, protected by **Reed-Solomon** error correction, and written at **pseudo-random pixel positions derived from the Kyber secret**. Every embed reports PSNR, SSIM and a chi-square histogram distance.

You send two files: the stego image and a small key file (`.kex`). Only the holder of the matching private key can find or read the message.

## Install

Needs Python 3.9 or newer. The easiest way is [pipx](https://pipx.pypa.io), which installs the `pqsteg` command in its own isolated environment:

```bash
pipx install git+https://github.com/YOUR-USERNAME/pqsteg
pqsteg --version
```

Without pipx, use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install git+https://github.com/YOUR-USERNAME/pqsteg
```

Installing from a downloaded copy of this repo works the same way: run `pipx install .` or `pip install .` inside the folder.

## Use it

**Receiver, once:** create a key pair and send the `.pub` file to the sender. Keep the `.key` file private.

```bash
pqsteg generate-keys --out-prefix alice
```

**Sender:** hide a message (any photo works as the cover; the output must be PNG or BMP).

```bash
pqsteg embed --cover photo.png --message "meet at 6pm" --public-key alice.pub --out stego.png
```

This writes `stego.png` and `stego.kex`. **Send both files**, as files and not as photos: JPEG conversion or chat-app recompression destroys the hidden data.

**Receiver:** recover it. The `.kex` file must sit next to the image with the same name.

```bash
pqsteg extract --stego stego.png --private-key alice.key
```

Other options: `--secret-file F` instead of `--message`; `--out F` on extract for binary data; `--bits 2` for 2 LSBs per channel (use the same on extract); `--privacy standard|enhanced` (below). `python -m pqsteg ...` works too.

## Privacy modes

In a terminal, `embed` asks:

```
Enhanced privacy mode hides at most 331 bytes of secret in this image (yours is 24 bytes).
It uses only 1% of the image, which is much harder to detect than standard mode.
Use enhanced privacy? [y/N]
```

The limit is calculated from your image: capacity is `W x H x 3 x bits / 8` bytes, enhanced mode may use 1 % of that, and the secret limit is what remains after the fixed overhead (28 bytes of encryption header plus about 33 % Reed-Solomon parity). Say yes with a secret within the limit and it embeds in enhanced mode; a secret that is too big stops with an error instead of quietly downgrading. Pass `--privacy standard` or `--privacy enhanced` to skip the question; without a terminal and without the flag, standard is used.

| Mode | Uses up to | Meaning |
|---|---|---|
| standard | 70 % of the image | large secrets; easier to detect statistically |
| enhanced | 1 % of the image | small secrets; near the noise floor of RS analysis |

**Why 1 %.** Random positions do not hide how much of the image was changed. RS analysis (a standard steganalysis test) on real photos, with an encrypted-looking payload at 1 bit per channel, estimated the fill almost exactly:

| photo | untouched | 1 % | 2 % | 5 % | 30 % |
|---|---|---|---|---|---|
| chelsea (451x300) | 0.0 % | 1.1 % | 2.0 % | 5.2 % | 31.1 % |
| coffee (600x400) | -0.5 % | 0.5 % | 1.4 % | 5.0 % | 30.0 % |

At 30 % that test reports 30 %: fully detected, and it even reveals the payload size. At 1 % its reading is within the noise of an untouched photo. This is three sample photos and one detector; stronger machine-learning detectors may still find LSB changes at 1 %. Treat enhanced mode as "much harder", not "undetectable".

## How it works

```
embed:    secret -> [A] encrypt -> [B] Reed-Solomon -> [C] random-position LSB -> stego.png
                \-> stego.kex                          \-> [D] PSNR / SSIM / chi-square
extract:  stego.kex + private key -> pixel order;  stego.png -> read bits -> repair -> decrypt
```

| File (in `pqsteg/`) | Role |
|---|---|
| `pqc_crypto.py` | **A** ML-KEM-768 key exchange, AES-256-GCM |
| `fec_engine.py` | **B** Reed-Solomon `encode` / `decode` |
| `steg_core.py` | **C** pixel order, embed / extract bits, PNG/BMP load/save |
| `metrics.py` | **D** PSNR, SSIM, chi-square |
| `pipeline.py` | `embed` / `extract` / `max_secret_bytes`; holds the two fill limits |
| `cli.py` | the `pqsteg` command |
| `exceptions.py` | error types, all deriving from `PQStegError` |

Each module docstring explains its design; read those first.

**The key exchange.** Each embed does one ML-KEM encapsulation against the receiver's public key. That gives a fresh 32-byte secret K and a 1088-byte "envelope" only the private key can open (the `.kex` file). K is split into an AES key for the payload and a position seed that decides which pixel bits carry it. Every message gets new randomness.

**Hidden in the image:** `[ length prefix 20 B ][ Reed-Solomon body ]`, which decodes to `nonce 12 | GCM tag 16 | encrypted secret`. Reed-Solomon RS(255,191) adds 64 parity bytes per block and repairs up to 32 wrong *bytes* per 255-byte block (12.5 %), roughly a 1.5 % random bit-error rate. `BODY_PARITY` in `fec_engine.py` trades capacity for robustness; sender and receiver must match.

**Which error means what**

| Exception | Meaning |
|---|---|
| `InvalidKeyError` | Malformed key or key file |
| `IntegrityError` | Hidden bits not found or not repairable: wrong private key or key file, wrong `--bits`, or the image was re-encoded |
| `DecryptionError` | Bits recovered fine, but the GCM tag failed: the hidden data was modified |
| `CapacityError` | Secret too big for the image or above the privacy mode's limit |
| `QualityError` | PSNR < 42 dB or SSIM < 0.98 |
| `ImageFormatError` | Unusable image, or a lossy output format (JPEG) |

A wrong private key gives `IntegrityError`, not `DecryptionError`: ML-KEM silently returns a wrong secret, which gives a wrong pixel order, so the receiver reads noise.

## Develop

```bash
git clone https://github.com/YOUR-USERNAME/pqsteg
cd pqsteg
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
mypy pqsteg tests
```

## Limits

1. **Two files must travel together.** Losing the `.kex` file means the message cannot be recovered. It reveals nothing without the private key.
2. **No robustness to processing.** JPEG recompression, resizing, cropping or screenshots destroy the data. Reed-Solomon only repairs in-place bit errors.
3. **`kyber-py` is pure Python, not constant-time and not audited.** For anything beyond a prototype, swap in liboqs (three calls in `pqc_crypto.py`).
4. **The sender is unauthenticated** (anyone with the public key can embed) and the private key file is stored unencrypted (mode 0600).
5. **No format versioning yet.** Files made by one release may not decode with another. Use the same version on both sides.
6. **Memory:** the pixel order costs about 23 bytes per slot; a 3-megapixel image takes ~2 s and ~290 MB. Downscale or crop very large photos.

## Licence

MIT, see [LICENSE](LICENSE).
