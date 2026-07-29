"""
Shared access to data/cutouts/*.fits - per-source(-group) MUSE cubelets with
DATA/STAT/MASK/CENTROIDS/PSF extensions, used to place lensed images at their
true (data-fit) sky position and to segment their true detected shape.

Every cutout shares the master data/muse-rgb.fits frame's CD matrix and
CRVAL - each cutout is just an integer-pixel crop of the same grid - so a
mask built in cutout-pixel space maps to muse-pixel space by a plain integer
index shift (verified: no sub-pixel remainder), with no reprojection needed.
"""
import glob
import os

from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
CUTOUTS_DIR = os.path.join(DATA_DIR, "cutouts")

MUSE_PIXEL_SCALE_ARCSEC = 0.2


def _muse_header():
    with fits.open(os.path.join(DATA_DIR, "muse-rgb.fits")) as hdul:
        return hdul[0].header.copy()


def compute_pixel_offset(cutout_hdr, muse_hdr):
    """Integer (dx, dy) such that muse_pixel[y+dy, x+dx] == cutout_pixel[y, x]
    (0-indexed). Both frames share CD/PC and CRVAL, so this is just the
    difference in CRPIX (FITS 1-indexed, cancels to a plain shift in
    0-indexed array coordinates)."""
    dx = muse_hdr["CRPIX1"] - cutout_hdr["CRPIX1"]
    dy = muse_hdr["CRPIX2"] - cutout_hdr["CRPIX2"]
    dxi, dyi = int(round(dx)), int(round(dy))
    if abs(dx - dxi) > 1e-6 or abs(dy - dyi) > 1e-6:
        raise ValueError("cutout is not pixel-aligned with the muse-rgb frame")
    return dxi, dyi


def iter_cutouts():
    """Yield (path, data, stat, mask, centroids_table, dx, dy) for every
    cutout file, dx/dy being its integer pixel offset into the muse-rgb
    frame."""
    muse_hdr = _muse_header()
    for path in sorted(glob.glob(os.path.join(CUTOUTS_DIR, "*.fits"))):
        with fits.open(path) as hdul:
            data = hdul["DATA"].data
            stat = hdul["STAT"].data
            mask = hdul["MASK"].data
            centroids = hdul["CENTROIDS"].data
            dx, dy = compute_pixel_offset(hdul["DATA"].header, muse_hdr)
        yield path, data, stat, mask, centroids, dx, dy


def load_lensed_image_centroids():
    """label (e.g. '8a') -> dict(ra, dec, pix_muse=(x, y), cutout_file=path).

    ra/dec come from the CENTROIDS table's own WCS-fit sky_centroid, i.e.
    fit directly from the data rather than transcribed from the paper -
    more accurate for the images that have one. pix_muse is the centroid's
    pixel position in the muse-rgb.fits grid."""
    out = {}
    for path, _data, _stat, _mask, centroids, dx, dy in iter_cutouts():
        for row in centroids:
            label = str(row["Source name"])
            cx, cy = row["pix_centroid"]
            out[label] = dict(
                ra=float(row["sky_centroid.ra"]),
                dec=float(row["sky_centroid.dec"]),
                pix_muse=(cx + dx, cy + dy),
                cutout_file=path,
            )
    return out
