"""
Build the cluster billboard image and real per-object cutout stamps.

- Cluster billboard: the full MUSE RGB frame, saved as-is (it is already a
  finished, linearly-scaled display image in data/muse-rgb.fits - a
  companion muse-rgb.pdf confirms it was prepared as a picture, not raw
  data, so we do NOT re-stretch it).
- Per-object stamps: a small cutout centered on each object's true RA/Dec,
  preferring the real MUSE RGB pixels (true color) and falling back to the
  HST F140W grayscale image when a position falls outside the MUSE
  footprint. A small neutral placeholder is used only for the handful of
  objects covered by neither image.
"""
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from PIL import Image

from prepare_data import load_field_catalog, load_lensed_sources

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")
STAMP_DIR = os.path.join(OUT_DIR, "stamps")

STAMP_RADIUS_ARCSEC = 2.5  # half-width of each cutout
FEATHER_FRAC = 0.35  # fraction of radius over which alpha falls off


def _wcs_2d_from_header(hdr):
    """Build a clean 2D WCS from a header that may have a stray NAXIS3."""
    hdr = hdr.copy()
    hdr["NAXIS"] = 2
    if "NAXIS3" in hdr:
        del hdr["NAXIS3"]
    return WCS(hdr, naxis=2)


def load_muse_rgb():
    with fits.open(os.path.join(DATA_DIR, "muse-rgb.fits")) as hdul:
        data = hdul[0].data  # (3, ny, nx), already in [0, 1]
        hdr = hdul[0].header
    rgb = np.moveaxis(data, 0, -1)  # (ny, nx, 3)
    rgb = np.clip(np.nan_to_num(rgb), 0.0, 1.0)
    wcs = _wcs_2d_from_header(hdr)
    return rgb, wcs


def load_hst():
    with fits.open(os.path.join(DATA_DIR, "hubble_f140w.fits")) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header
    data = np.nan_to_num(data)
    lo, hi = np.percentile(data, [1, 99.7])
    stretched = np.arcsinh(np.clip((data - lo) / (hi - lo + 1e-12), 0, None) * 10) / np.arcsinh(10)
    stretched = np.clip(stretched, 0, 1)
    gray_rgb = np.stack([stretched] * 3, axis=-1)
    wcs = WCS(hdr)
    return gray_rgb, wcs


def save_billboard(rgb):
    img = (rgb * 255).astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(os.path.join(OUT_DIR, "muse_rgb.png"))


def _feathered_alpha(size):
    """Circular alpha mask, feathered near the edge, for a size x size stamp."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2.0)
    inner = 1.0 - FEATHER_FRAC
    alpha = np.clip((1.0 - r) / FEATHER_FRAC, 0, 1)
    alpha[r <= inner] = 1.0
    alpha[r >= 1.0] = 0.0
    return alpha


def extract_stamp(ra_deg, dec_deg, muse_rgb, muse_wcs, hst_rgb, hst_wcs, radius_arcsec=STAMP_RADIUS_ARCSEC):
    """Return (rgba stamp as float [0,1] array, source) where source is
    'muse', 'hst', or 'none'."""
    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)

    for rgb, wcs, name, scale_arcsec_per_px in (
        (muse_rgb, muse_wcs, "muse", 0.2),
        (hst_rgb, hst_wcs, "hst", 0.07),
    ):
        px, py = wcs.wcs_world2pix([[ra_deg, dec_deg]], 0)[0]
        half = int(round(radius_arcsec / scale_arcsec_per_px))
        ny, nx = rgb.shape[:2]
        x0, x1 = int(round(px)) - half, int(round(px)) + half
        y0, y1 = int(round(py)) - half, int(round(py)) + half
        if x0 < 0 or y0 < 0 or x1 >= nx or y1 >= ny:
            continue
        crop = rgb[y0 : y1 + 1, x0 : x1 + 1, :]
        if crop.shape[0] < 3 or crop.shape[1] < 3:
            continue
        alpha = _feathered_alpha(crop.shape[0])
        if name == "muse":
            # Per-stamp auto-exposure + gamma: the same physical galaxy can
            # be a bright cluster member or a barely-detected z>3 Lyman-alpha
            # emitter, so a single global stretch either blows out the
            # bright ones or leaves the faint ones invisible. Normalize each
            # stamp by its own bright-pixel level (not its absolute peak, to
            # avoid amplifying single-pixel noise), preserving relative
            # color, then gamma-stretch. This is a per-object "exposure"
            # choice, not a reshaping of the data - real colors, real shapes.
            # The full muse_rgb.png billboard is untouched.
            norm = max(np.percentile(crop, 99.0), 1e-3)
            boosted = np.clip(crop / norm, 0, 1) ** 0.42
        else:
            boosted = crop
        rgba = np.dstack([boosted, alpha])
        return rgba, name, radius_arcsec

    # neither image covers this position
    size = 9
    alpha = _feathered_alpha(size) * 0.6
    rgb_fill = np.zeros((size, size, 3))
    rgb_fill[..., 2] = 0.35  # dim blue-gray placeholder
    rgb_fill[..., 0] = 0.15
    rgb_fill[..., 1] = 0.2
    rgba = np.dstack([rgb_fill, alpha])
    return rgba, "none", radius_arcsec


def build_all_stamps():
    os.makedirs(STAMP_DIR, exist_ok=True)
    muse_rgb, muse_wcs = load_muse_rgb()
    hst_rgb, hst_wcs = load_hst()
    save_billboard(muse_rgb)

    field_rows = load_field_catalog()
    lensed_rows = load_lensed_sources()

    stamps = {}
    counts = {"muse": 0, "hst": 0, "none": 0}
    for r in field_rows + lensed_rows:
        rgba, source, radius = extract_stamp(r["ra"], r["dec"], muse_rgb, muse_wcs, hst_rgb, hst_wcs)
        key = r["label"] if "source" not in r else f"img_{r['label']}"
        stamps[key] = dict(rgba=rgba, source=source, radius_arcsec=radius)
        counts[source] += 1

    print("Stamp sources:", counts)

    # dump a handful to disk for visual inspection
    sample_keys = list(stamps.keys())[:6] + [k for k in stamps if k.startswith("img_")][:6]
    for k in dict.fromkeys(sample_keys):
        rgba = stamps[k]["rgba"]
        img = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img, mode="RGBA").save(os.path.join(STAMP_DIR, f"{k}.png"))

    np.save(os.path.join(OUT_DIR, "stamps.npy"), stamps, allow_pickle=True)
    return stamps


if __name__ == "__main__":
    build_all_stamps()
    print(f"Saved billboard to {os.path.join(OUT_DIR, 'muse_rgb.png')}")
    print(f"Saved stamp cache to {os.path.join(OUT_DIR, 'stamps.npy')}")
    print(f"Sample stamp PNGs in {STAMP_DIR}")
