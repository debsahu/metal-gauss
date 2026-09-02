#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26", "Pillow>=10", "tifffile>=2024.1", "imagecodecs>=2024.1",
# ]
# ///
"""The ONE implementation of Brush's depth/normal prior wire formats, Python side.

Three formats, one decoded contract. Whatever the bytes on disk look like, a
reader gets back exactly what Brush's loaders hand the trainer:

    depth   float32 (H, W)      metres, 0 == invalid
    normal  float32 (H, W, 3)   unit, camera-frame OpenCV, TOWARD-camera,
                                (0,0,0) == invalid

    "tiff-f32"          today's file: uncompressed float32 TIFF. Byte-for-byte
                        what every writer in this directory emitted before this
                        module existed -- pinned by a test, because "the default
                        changes nothing" is the only reason it is still default.
    "tiff-f32-deflate"  same float32 samples, Deflate + TIFF FloatingPoint
                        predictor (3). LOSSLESS: the predictor is a byte
                        transpose plus a delta, so the u32 bit pattern of every
                        sample survives. Measured 2.3x on 4K depth, 1.4x on
                        normals -- float32 mantissa noise is close to
                        incompressible, so do not expect more.
    "png-quantized"     uint16-millimetre grayscale PNG (depth) / uint8 RGB PNG
                        (normals). 15-40x smaller, and lossy by a bounded,
                        MEASURED amount: <= 0.5 mm + O(ulp) of depth, and <= 2/255
                        per normal component -- not the 1/255 the plan's prose
                        gives, because the code-128 override moves a component of
                        up to 2/255 all the way to zero. Only code 128 can exceed
                        a half step; measured 976 of 480,000 random components,
                        max 0.00783.

WHY THE QUANTIZED CODEC IS NOT OURS. It is `gauss-surf`'s (Pablo Vela,
`rerun-io/examples-monorepo`, Apache-2.0), copied deliberately and exactly:

    depth encode   packages/gauss-surf/src/gauss_surf/uw_geometry.py:234-238
    depth decode   packages/gauss-surf/src/gauss_surf/render_io.py:146-161
    normal codec   packages/gauss-surf/src/gauss_surf/normals_encoding.py:44,52-53

Adopting it verbatim means his golden vectors, his cache blobs and his tests
transfer to us unchanged. The alternative -- a "cleaner" symmetric map -- would
buy nothing and cost a silent cross-implementation disagreement on two codes,
forever. Two consequences of that decision are load-bearing and easy to
"fix" wrongly, so they are spelled out:

  * NORMAL CODE 128 IS SPECIAL-CASED TO EXACTLY 0.0, and nothing else is. So
    the decode is NOT symmetric about the middle: code 127 -> -1/255 but code
    129 -> +3/255. That asymmetry is deliberate. It is bounded by 2/255, i.e.
    the same order as the quantization floor, and it is the price of letting the
    (0,0,0)-invalid sentinel round-trip exactly. Any uint8 map that represents
    both +-1 and an exact 0 has a wart somewhere; this is where ours is.
  * DEPTH ROUNDING IS numpy's `rint`, i.e. HALF-TO-EVEN. 0.5 mm rounds DOWN to
    code 0, which is the invalid sentinel -- a half-millimetre depth is thrown
    away. Deliberate, matches the reference, pinned by a test, and irrelevant
    to any scene we shoot (measured minimum depth 0.579 m).

SIGN CONVENTION IS NOT PART OF THE CODEC. `gauss-surf` stores normals pointing
AWAY from the camera; Brush's loader contract (crates/brush-dataset/src/
load_normal.rs) is TOWARD the camera, and every prior we have on disk is
toward-camera (measured: fraction with n_z < 0 among valid = 0.9996 playroom,
1.0000 ARKit). Our PNGs therefore store toward-camera normals. Importing a
foreign bundle is an extraction-time negation (`convert_priors.py
--negate-normals`), never a decode-time one: a loader that guesses provenance is
a loader that can be wrong silently.

NO RESIZING, EVER. Brush hard-asserts that a prior matches the image size it
loaded (`load_depth.rs`, `load_normal.rs`). Note in passing that the reference's
cache bilinearly downscales depth INCLUDING the 0 sentinel, which smears
invalid into valid; that cannot happen here because nothing here resizes, and
nothing here should learn to.
"""
from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Final

import numpy as np

__all__ = [
    "PRIOR_FORMATS",
    "PriorFormatError",
    "decode_depth_u16mm",
    "decode_normal_u8",
    "detect_format",
    "find_prior",
    "list_priors",
    "encode_depth_u16mm",
    "encode_normal_u8",
    "prior_path",
    "read_depth",
    "read_normal",
    "write_depth",
    "write_normal",
    "write_tiff_array",
]

PRIOR_FORMATS: Final = ("tiff-f32", "tiff-f32-deflate", "png-quantized")

#: Little- and big-endian TIFF magics, and the PNG signature. Format is decided
#: by these bytes and never by the file extension -- Brush's prior discovery is
#: stem-based and extension-blind, so `.tif`, `.tiff` and `.png` all reach the
#: same code path and the bytes are the only honest discriminator.
_TIFF_MAGICS: Final = (b"II*\x00", b"MM\x00*")
_PNG_MAGIC: Final = b"\x89PNG\r\n\x1a\n"
_MAGIC_LEN: Final = 8

#: Depth codes are millimetres in a uint16, so the representable range ends here.
#: Deeper-than-this stays VALID and saturates (reference behaviour); it does not
#: fall back to the invalid sentinel.
DEPTH_MM_MAX: Final = 65535.0

#: Normal components must lie in [-1, 1]. Renormalisation upstream can leave a
#: component a few ulps outside; clip that away, but refuse anything larger --
#: `rint((1.01 + 1) / 2 * 255) = 256`, which wraps to code 0 in uint8 and turns a
#: perfectly good normal into the invalid sentinel without a word.
NORMAL_RANGE_ATOL: Final = 1e-4


class PriorFormatError(ValueError):
    """A prior file is not one of the accepted wire formats, or is the wrong one.

    Deliberately a hard error rather than a best-effort conversion. Every silent
    coercion available here (8-bit depth read as 16, RGBA normals squashed to
    RGB, float64 samples truncated to float32) produces a file that trains, just
    not on the data anybody intended.
    """


# --------------------------------------------------------------------------- #
# codecs -- pure array functions, no I/O                                       #
# --------------------------------------------------------------------------- #

def encode_depth_u16mm(depth_m: np.ndarray) -> np.ndarray:
    """Metric float32 depth -> uint16 millimetres. Ref: `uw_geometry.py:234-238`.

    Invalid is anything non-finite or <= 0, and it encodes to code 0. Valid
    depth beyond 65.535 m saturates to 65535 and STAYS valid.
    """
    depth_m = np.asarray(depth_m)
    if depth_m.ndim != 2:
        raise PriorFormatError(f"depth must be 2-D (H, W), got shape {depth_m.shape}")
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    safe_m = np.where(valid, depth_m, 0.0).astype(np.float32, copy=False)
    depth_mm_float = np.clip(safe_m * 1000.0, 0.0, DEPTH_MM_MAX)
    return np.rint(depth_mm_float).astype(np.uint16)


def decode_depth_u16mm(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 millimetres -> float32 metres. Ref: `render_io.py:146-161`.

    The op order (`f32(mm) / 1000.0`) is pinned so that Rust's
    `(mm as f32) / 1000.0` is bit-identical on every one of the 65536 codes, not
    merely on the anchors.
    """
    depth_mm = np.asarray(depth_mm)
    if depth_mm.dtype != np.uint16:
        raise PriorFormatError(f"depth codes must be uint16, got {depth_mm.dtype}")
    if depth_mm.ndim != 2:
        raise PriorFormatError(f"depth codes must be 2-D (H, W), got shape {depth_mm.shape}")
    return (depth_mm.astype(np.float32) / 1000.0).astype(np.float32, copy=False)


def encode_normal_u8(normal: np.ndarray) -> np.ndarray:
    """Signed unit normals -> uint8 RGB codes. Ref: `normals_encoding.py:44`.

    `(0,0,0)` invalid maps to code 128 in every channel (`rint(127.5) == 128`,
    half-to-even), which the decoder maps back to exactly `(0,0,0)`.
    """
    normal = np.asarray(normal)
    if normal.ndim != 3 or normal.shape[-1] != 3:
        raise PriorFormatError(f"normals must have shape (H, W, 3), got {normal.shape}")
    if not np.all(np.isfinite(normal)):
        raise PriorFormatError("normals must contain only finite values")
    lo, hi = float(np.min(normal)), float(np.max(normal))
    if lo < -1.0 - NORMAL_RANGE_ATOL or hi > 1.0 + NORMAL_RANGE_ATOL:
        raise PriorFormatError(
            f"normal components must lie within [-1, 1] (+-{NORMAL_RANGE_ATOL:g} "
            f"of float noise), got [{lo:.6g}, {hi:.6g}]"
        )
    clipped = np.clip(normal, -1.0, 1.0).astype(np.float32, copy=False)
    return np.rint((clipped + 1.0) / 2.0 * 255.0).astype(np.uint8)


def decode_normal_u8(codes: np.ndarray) -> np.ndarray:
    """uint8 RGB codes -> signed float32 normals. Ref: `normals_encoding.py:52-53`.

    Code 128 -- and ONLY code 128 -- decodes to exactly 0.0, per channel. See the
    module docstring on why 127 and 129 are therefore not symmetric about it.
    """
    codes = np.asarray(codes)
    if codes.dtype != np.uint8:
        raise PriorFormatError(f"normal codes must be uint8, got {codes.dtype}")
    if codes.ndim != 3 or codes.shape[-1] != 3:
        raise PriorFormatError(f"normal codes must have shape (H, W, 3), got {codes.shape}")
    decoded = codes.astype(np.float32) / 255.0 * 2.0 - 1.0
    decoded[codes == 128] = 0.0
    return decoded


# --------------------------------------------------------------------------- #
# format detection                                                             #
# --------------------------------------------------------------------------- #

def _sniff(data: bytes, where: str) -> str:
    """'tiff' or 'png' from the leading magic bytes, else a hard error."""
    if data[:4] in _TIFF_MAGICS:
        return "tiff"
    if data[:8] == _PNG_MAGIC:
        return "png"
    raise PriorFormatError(
        f"{where}: unsupported prior format (expected float32 TIFF magic "
        f"II*\\0 / MM\\0* or PNG magic \\x89PNG\\r\\n\\x1a\\n, got {data[:8]!r})"
    )


def _png_ihdr(data: bytes, where: str) -> tuple[int, int, int, int, int]:
    """(width, height, bit_depth, colour_type, interlace) straight out of IHDR.

    Read from the bytes rather than asked of Pillow on purpose: Pillow's `mode`
    strings fold distinctions we must not fold (an 8-bit grayscale depth map and
    a 16-bit one both become usable arrays), and the fork's loader matches on the
    exact decoded variant. Checking IHDR keeps the two languages strict in the
    same place.
    """
    if len(data) < 33 or data[12:16] != b"IHDR":
        raise PriorFormatError(f"{where}: PNG has no IHDR chunk where one must be")
    width, height, bit_depth, colour, _compression, _filter, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    return width, height, bit_depth, colour, interlace


def detect_format(path: str | Path) -> str:
    """One of :data:`PRIOR_FORMATS`, decided by content.

    A TIFF is reported as ``tiff-f32-deflate`` when its first page carries any
    compression, and ``tiff-f32`` when it does not.
    """
    import tifffile

    path = Path(path)
    with open(path, "rb") as handle:
        magic = handle.read(_MAGIC_LEN)
    kind = _sniff(magic, str(path))
    if kind == "png":
        return "png-quantized"
    with tifffile.TiffFile(path) as tif:
        return "tiff-f32" if int(tif.pages[0].compression) == 1 else "tiff-f32-deflate"


def prior_path(path: str | Path, fmt: str) -> Path:
    """The path a prior of `fmt` actually lands on, given any candidate path.

    Quantized priors are `<stem>.png`, float32 ones `<stem>.tiff`. Brush finds
    priors by stem and never looks at the extension, so this is a local naming
    choice and not a wire contract -- but it does mean a converted dataset must
    have the old file removed, or discovery finds two files for one stem.

    Only a KNOWN prior extension is replaced; anything else is appended to. Our
    real frame names carry dots (`48018538_828208.786.tiff`), so a bare
    `Path.with_suffix` on an extensionless name would eat `.786` and silently
    rename the frame -- which discovery would then simply not find.
    """
    if fmt not in PRIOR_FORMATS:
        raise PriorFormatError(f"unknown prior format {fmt!r}, expected one of {PRIOR_FORMATS}")
    wanted = ".png" if fmt == "png-quantized" else ".tiff"
    candidate = Path(path)
    if candidate.suffix.lower() in (".png", ".tif", ".tiff"):
        return candidate.with_suffix(wanted)
    return candidate.with_name(candidate.name + wanted)


# --------------------------------------------------------------------------- #
# file I/O                                                                     #
# --------------------------------------------------------------------------- #

def write_tiff_array(path: str | Path, arr: np.ndarray, fmt: str) -> None:
    """Write a float array as a TIFF in `fmt`, pinning the 3-channel photometric.

    Public because `recompress_priors.py` needs the same pinning; it is the one
    place TIFF encoder options are chosen.
    """
    import tifffile

    # tifffile currently stores a contiguous (H, W, 3) float32 array as RGB, but
    # warns that a future version will default it to MINISBLACK in separate pages.
    # That flip would silently produce normal TIFFs the fork's 3-channel decoder
    # cannot read, so the current behaviour is pinned explicitly. Verified
    # byte-identical to the unpinned call today, which is what keeps
    # `tiff-f32` inert.
    extra = {"photometric": "rgb"} if (arr.ndim == 3 and arr.shape[-1] == 3) else {}
    if fmt == "tiff-f32":
        tifffile.imwrite(path, arr, **extra)
    else:
        # predictor=True on float samples selects TIFF predictor 3 (FloatingPoint),
        # which the Rust `tiff` crate decodes natively -- that is the entire reason
        # this is deflate+predictor and not zstd.
        tifffile.imwrite(path, arr, compression="deflate", predictor=True, **extra)


def write_depth(path: str | Path, depth_m: np.ndarray, fmt: str = "tiff-f32") -> Path:
    """Write metric depth, returning the path actually written."""
    if fmt not in PRIOR_FORMATS:
        raise PriorFormatError(f"unknown prior format {fmt!r}, expected one of {PRIOR_FORMATS}")
    depth_m = np.asarray(depth_m)
    if depth_m.ndim != 2:
        raise PriorFormatError(f"depth must be 2-D (H, W), got shape {depth_m.shape}")
    out = prior_path(path, fmt)
    if fmt == "png-quantized":
        from PIL import Image

        Image.fromarray(encode_depth_u16mm(depth_m)).save(out, format="PNG")
    else:
        write_tiff_array(out, np.asarray(depth_m, dtype=np.float32), fmt)
    return out


def write_normal(path: str | Path, normal: np.ndarray, fmt: str = "tiff-f32") -> Path:
    """Write unit toward-camera normals, returning the path actually written."""
    if fmt not in PRIOR_FORMATS:
        raise PriorFormatError(f"unknown prior format {fmt!r}, expected one of {PRIOR_FORMATS}")
    normal = np.asarray(normal)
    if normal.ndim != 3 or normal.shape[-1] != 3:
        raise PriorFormatError(f"normals must have shape (H, W, 3), got {normal.shape}")
    out = prior_path(path, fmt)
    if fmt == "png-quantized":
        from PIL import Image

        Image.fromarray(encode_normal_u8(normal), mode="RGB").save(out, format="PNG")
    else:
        write_tiff_array(out, np.asarray(normal, dtype=np.float32), fmt)
    return out


#: Extensions a prior may carry on disk, in the order discovery reports them.
PRIOR_SUFFIXES: Final = (".tiff", ".tif", ".png")


def list_priors(directory: str | Path) -> list[Path]:
    """Every prior file in `directory`, whatever its extension, sorted by stem.

    Readers must glob this rather than `*.tiff`, or a migrated dataset silently
    looks empty and the tool reports "0 frames" instead of failing.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir()
         if p.is_file() and not p.name.startswith(".")
         and p.suffix.lower() in PRIOR_SUFFIXES),
        key=lambda p: (p.stem, p.suffix),
    )


def find_prior(directory: str | Path, stem: str) -> Path | None:
    """The one prior file for `stem`, or None. TWO matches is a hard error.

    Mirrors the fork's discovery contract (plan D4): a dataset mid-migration can
    hold `x.tiff` beside `x.png`, and picking either one silently is how a run
    ends up trained on stale supervision. The failure names both paths.
    """
    directory = Path(directory)
    matches = [directory / f"{stem}{suffix}" for suffix in PRIOR_SUFFIXES
               if (directory / f"{stem}{suffix}").exists()]
    if not matches:
        return None
    if len(matches) > 1:
        raise PriorFormatError(
            "ambiguous prior: found " + " and ".join(str(m) for m in matches)
            + f" for stem {stem!r}. A migration was interrupted -- delete the stale one."
        )
    return matches[0]


def _read_bytes(path: Path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def read_depth(path: str | Path) -> np.ndarray:
    """Read any supported depth prior as float32 metres, 0 == invalid."""
    import tifffile

    path = Path(path)
    data = _read_bytes(path)
    if _sniff(data, str(path)) == "tiff":
        arr = tifffile.imread(io.BytesIO(data))
        if arr.dtype != np.float32:
            raise PriorFormatError(
                f"{path}: depth TIFF must hold float32 samples, got {arr.dtype}"
            )
        if arr.ndim != 2:
            raise PriorFormatError(f"{path}: depth TIFF must be single-channel, got {arr.shape}")
        return np.ascontiguousarray(arr)

    from PIL import Image

    _w, _h, bit_depth, colour, interlace = _png_ihdr(data, str(path))
    if (bit_depth, colour) != (16, 0):
        raise PriorFormatError(
            f"{path}: depth PNG must be 16-bit grayscale (bit depth 16, colour type 0), "
            f"got bit depth {bit_depth}, colour type {colour}"
        )
    if interlace:
        raise PriorFormatError(f"{path}: interlaced depth PNG is not supported")
    with Image.open(io.BytesIO(data)) as image:
        codes = np.asarray(image, dtype=np.uint16)
    return decode_depth_u16mm(codes)


def read_normal(path: str | Path) -> np.ndarray:
    """Read any supported normal prior as float32 unit normals, (0,0,0) invalid."""
    import tifffile

    path = Path(path)
    data = _read_bytes(path)
    if _sniff(data, str(path)) == "tiff":
        arr = tifffile.imread(io.BytesIO(data))
        if arr.dtype != np.float32:
            raise PriorFormatError(
                f"{path}: normal TIFF must hold float32 samples, got {arr.dtype}"
            )
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise PriorFormatError(
                f"{path}: normal TIFF must be 3-channel (H, W, 3), got {arr.shape}"
            )
        return np.ascontiguousarray(arr)

    from PIL import Image

    _w, _h, bit_depth, colour, interlace = _png_ihdr(data, str(path))
    if (bit_depth, colour) != (8, 2):
        raise PriorFormatError(
            f"{path}: normal PNG must be 8-bit RGB (bit depth 8, colour type 2), "
            f"got bit depth {bit_depth}, colour type {colour}"
        )
    if interlace:
        raise PriorFormatError(f"{path}: interlaced normal PNG is not supported")
    with Image.open(io.BytesIO(data)) as image:
        codes = np.asarray(image, dtype=np.uint8)
    return decode_normal_u8(codes)
