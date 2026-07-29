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
from scipy import ndimage
import astropy.units as u
from PIL import Image

from prepare_data import load_field_catalog, load_lensed_sources
from cutout_data import iter_cutouts, MUSE_PIXEL_SCALE_ARCSEC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")
STAMP_DIR = os.path.join(OUT_DIR, "stamps")

STAMP_RADIUS_ARCSEC = 2.5  # half-width of each cutout
FEATHER_FRAC = 0.35  # fraction of radius over which alpha falls off

# --- lensed-image arc shapes, from data/cutouts/*.fits ---------------------
# Each cutout carries DATA/STAT/MASK/CENTROIDS for one source (or small
# group of sources). We segment each image's true detected footprint from
# its own S/N map instead of stamping it into a circle, so arcs render as
# arcs. See cutout_data.py for the pixel-grid alignment this relies on.
ARC_SN_THRESHOLD = 2.0  # S/N above which a pixel counts as "the source"
ARC_FEATHER_SIGMA_PX = 0.8  # gaussian blur applied to the binary mask edge
ARC_BBOX_PAD_PX = 4  # margin around the detected footprint, for feathering
ARC_BLOB_SEARCH_RADIUS_PX = 5  # how far to look if a centroid pixel itself
# falls just under threshold (confirmed needed for 3 of 41 images)

# --- stamp exposure --------------------------------------------------------
# A stamp can be a bright cluster member or a barely-detected z>3 Lyman-alpha
# emitter, so the faint end needs lifting or it is invisible next to the
# bright end. But the lift has to leave the stamp looking like the MUSE
# picture it was cut from: the cluster billboard cross-dissolves into these
# stamps in flythrough.py, and that only reads as a dissolve if the pixels
# agree.
#
# So the lift is a single scalar gain applied equally to R, G and B -- hue and
# saturation are preserved exactly -- bounded below at 1.0 so anything already
# at full scale passes through identical to the billboard. Measured p99
# luminance over the 78 objects inside the MUSE footprint runs 0.09 to 1.00
# with a median of 0.81, so most stamps come through untouched and only the
# faintest arcs are really boosted.
#
# The previous version applied a per-channel `(crop / p99) ** 0.42`. That
# renormalised even already-bright galaxies up toward white, and desaturated
# everything, because (R/G)**0.42 pulls every channel ratio toward 1.
EXPOSURE_TARGET = 0.85  # bright level a faint stamp is lifted toward
EXPOSURE_MAX_GAIN = 8.0  # cap, so the faintest arcs don't just amplify noise


def apply_exposure(crop):
    """Chroma-exact exposure lift: scale R, G and B by one bounded scalar."""
    lum = crop.max(axis=-1)
    p99 = float(np.percentile(lum, 99.0))
    gain = float(np.clip(EXPOSURE_TARGET / max(p99, 1e-3), 1.0, EXPOSURE_MAX_GAIN))
    return np.clip(crop * gain, 0.0, 1.0)


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
        boosted = apply_exposure(crop) if name == "muse" else crop
        rgba = np.dstack([boosted, alpha])
        return rgba, name, radius_arcsec

    # neither image covers this position - a dim, small, low-contrast fill
    # so the ~17 uncovered objects recede into the background rather than
    # reading as a prominent blob/artifact next to real imagery
    size = 9
    alpha = _feathered_alpha(size) * 0.25
    rgb_fill = np.full((size, size, 3), 0.12)
    rgba = np.dstack([rgb_fill, alpha])
    return rgba, "none", radius_arcsec * 0.6


def _nearest_labeled_pixel(labeled, iy, ix, max_r=ARC_BLOB_SEARCH_RADIUS_PX):
    """labeled[iy, ix] if nonzero, else the label of the nearest nonzero
    pixel within max_r (a handful of centroids sit a pixel or two below the
    S/N threshold - confirmed true for 3 of 41 images)."""
    if labeled[iy, ix] != 0:
        return labeled[iy, ix]
    ny, nx = labeled.shape
    for r in range(1, max_r + 1):
        y0, y1 = max(0, iy - r), min(ny - 1, iy + r)
        x0, x1 = max(0, ix - r), min(nx - 1, ix + r)
        sub = labeled[y0 : y1 + 1, x0 : x1 + 1]
        ys, xs = np.where(sub > 0)
        if len(ys):
            yy, xx = ys + y0, xs + x0
            d2 = (yy - iy) ** 2 + (xx - ix) ** 2
            k = np.argmin(d2)
            return sub[ys[k], xs[k]]
    return 0


def _image_sort_key(label):
    """'12b' -> (12, 'b'), so a blob's members come out in catalog order and
    the primary member is deterministic."""
    digits = "".join(ch for ch in label if ch.isdigit())
    letters = "".join(ch for ch in label if not ch.isdigit())
    return (int(digits) if digits else 0, letters)


def build_lensed_image_masks(muse_rgb, muse_wcs):
    """primary label ('8a') -> dict(rgba, half_width_arcsec,
    half_height_arcsec, merged_labels, ra, dec) for every connected
    lensed-image footprint that has a data/cutouts/*.fits cubelet, segmenting
    its true detected shape instead of stamping it into a circle.

    Approach per cutout file: threshold the S/N map (DATA/sqrt(STAT), with
    MASK==0 pixels - hot pixels / unrelated contaminating sources - zeroed
    out first) and connected-component label it. One stamp per connected
    blob: where several catalogued images share a blob (confirmed for
    {3a,3b,3c}, {5a,5b}, {9a,9b}, {12a,12b}) they are contiguous in the data
    and are drawn as one footprint, keyed by the first of them, with the rest
    listed in `merged_labels` so the renderer can drop their duplicate catalog
    rows. An earlier version split these by nearest centroid, which cut one
    real blob into arbitrary pieces.

    This is a statement about footprints, not about how many images there are:
    the field still has 41 lensed images, some of which happen to touch, the
    same way an Einstein ring is still four images.

    `ra`/`dec` are the sky position of the blob's *bounding-box centre*, which
    is what the renderer must centre the stamp on. It is not the same point as
    any catalog centroid, so positioning an arc stamp on its centroid (as this
    originally did) offsets the pixels from where they belong."""
    out = {}
    for path, data, stat, mask, centroids, dx, dy in iter_cutouts():
        sn = np.where(mask == 1, data / np.sqrt(stat), 0.0)
        labeled, _n = ndimage.label(sn > ARC_SN_THRESHOLD)

        pix_centroid = {
            str(row["Source name"]): row["pix_centroid"] for row in centroids
        }
        blob_id = {
            name: _nearest_labeled_pixel(labeled, int(round(cy)), int(round(cx)))
            for name, (cx, cy) in pix_centroid.items()
        }

        by_blob = {}
        for name, bid in blob_id.items():
            if bid == 0:
                # no thresholded pixel anywhere near this centroid; leave the
                # image on the circular-stamp fallback rather than handing it
                # `labeled == 0`, which is the entire background
                print(f"  no arc blob found for {name} ({os.path.basename(path)})")
                continue
            by_blob.setdefault(bid, []).append(name)

        ny, nx = labeled.shape
        for bid, names in by_blob.items():
            names = sorted(names, key=_image_sort_key)
            ys, xs = np.where(labeled == bid)
            y0 = max(0, ys.min() - ARC_BBOX_PAD_PX)
            y1 = min(ny - 1, ys.max() + ARC_BBOX_PAD_PX)
            x0 = max(0, xs.min() - ARC_BBOX_PAD_PX)
            x1 = min(nx - 1, xs.max() + ARC_BBOX_PAD_PX)

            local_mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
            local_mask[ys - y0, xs - x0] = 1.0
            alpha = ndimage.gaussian_filter(local_mask, ARC_FEATHER_SIGMA_PX)
            alpha = np.clip(alpha / max(alpha.max(), 1e-9), 0, 1)

            my0, my1 = y0 + dy, y1 + dy
            mx0, mx1 = x0 + dx, x1 + dx
            crop = muse_rgb[my0 : my1 + 1, mx0 : mx1 + 1, :]
            boosted = apply_exposure(crop)

            ra, dec = muse_wcs.wcs_pix2world(
                [[(mx0 + mx1) / 2.0, (my0 + my1) / 2.0]], 0
            )[0]
            out[names[0]] = dict(
                rgba=np.dstack([boosted, alpha]),
                half_width_arcsec=(x1 - x0 + 1) / 2.0 * MUSE_PIXEL_SCALE_ARCSEC,
                half_height_arcsec=(y1 - y0 + 1) / 2.0 * MUSE_PIXEL_SCALE_ARCSEC,
                merged_labels=names,
                ra=float(ra),
                dec=float(dec),
            )
    return out


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
        stamps[key] = dict(
            rgba=rgba, source=source,
            half_width_arcsec=radius, half_height_arcsec=radius,
        )
        counts[source] += 1

    # Lensed images with a data/cutouts/*.fits cubelet get their true
    # detected (generally non-circular) shape instead of the circular stamp
    # above - overrides the img_<label> entries just written. Images sharing
    # one connected footprint collapse to a single stamp under the first of
    # them; the others are dropped from the cache entirely so the renderer
    # can't draw the same pixels twice.
    arc_masks = build_lensed_image_masks(muse_rgb, muse_wcs)
    counts["muse_arc"] = 0
    n_merged_away = 0
    for label, arc in arc_masks.items():
        key = f"img_{label}"
        if key not in stamps:
            continue  # e.g. img_8d: has a cutout but isn't in the catalog (no confirmed z)
        members = [m for m in arc["merged_labels"] if f"img_{m}" in stamps]
        counts[stamps[key]["source"]] -= 1
        stamps[key] = dict(
            rgba=arc["rgba"], source="muse_arc",
            half_width_arcsec=arc["half_width_arcsec"],
            half_height_arcsec=arc["half_height_arcsec"],
            merged_labels=members,
            ra=arc["ra"], dec=arc["dec"],
        )
        counts["muse_arc"] += 1
        for m in members[1:]:
            counts[stamps.pop(f"img_{m}")["source"]] -= 1
            n_merged_away += 1
        if len(members) > 1:
            print(f"  contiguous footprint: {' + '.join(members)} -> one stamp")

    n_images = sum(1 for r in lensed_rows)
    print("Stamp sources:", counts)
    print(
        f"{n_images} lensed images -> {n_images - n_merged_away} stamps "
        f"({n_merged_away} share a footprint with another image)"
    )

    # dump a handful to disk for visual inspection, including a few
    # high-elongation arcs to confirm shape (not circular) and clean edges
    sample_keys = (
        list(stamps.keys())[:6]
        + [k for k in stamps if k.startswith("img_")][:6]
        + [f"img_{lbl}" for lbl in ("8a", "12a", "6d") if f"img_{lbl}" in stamps]
    )
    for k in dict.fromkeys(sample_keys):
        rgba = stamps[k]["rgba"]
        # flipud only for these human-eyeball PNGs: the cached arrays stay in
        # FITS row order (row 0 = lowest Dec) and the renderer draws them with
        # origin="lower", but a PNG viewer puts row 0 at the top, which would
        # show every spot-check upside down.
        img = (np.clip(rgba, 0, 1)[::-1] * 255).astype(np.uint8)
        Image.fromarray(img, mode="RGBA").save(os.path.join(STAMP_DIR, f"{k}.png"))

    np.save(os.path.join(OUT_DIR, "stamps.npy"), stamps, allow_pickle=True)
    return stamps


if __name__ == "__main__":
    build_all_stamps()
    print(f"Saved billboard to {os.path.join(OUT_DIR, 'muse_rgb.png')}")
    print(f"Saved stamp cache to {os.path.join(OUT_DIR, 'stamps.npy')}")
    print(f"Sample stamp PNGs in {STAMP_DIR}")
