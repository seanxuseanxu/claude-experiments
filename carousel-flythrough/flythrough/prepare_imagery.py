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
- Lensed images get their true segmented footprint from data/cutouts/*.fits
  rather than a circle: broadband pixels for the continuum-detected arcs, and
  the continuum-subtracted line map for the Lyman-alpha emitters and 4e,
  which have no broadband light of their own. See LAE_SOURCES.
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
from lensed_sources import LENSED_IMAGES, source_label
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
# falls just under threshold (confirmed needed for 3 of the 40 images)

# --- line-map stamps: the Lyman-alpha emitters, and 4e ---------------------
# Sources 8, 11, 12 and 13 are the field's four LAEs, and Carousel Lens I says
# three of them "are not apparent in imaging data and are only detected in
# MUSE IFU data". So sampling their colour out of muse-rgb.fits - a broadband
# picture in which they are, by definition, absent - samples whatever else
# happens to lie along that sightline. That produced two failure modes: an
# image buried in a cluster galaxy's light came out gold (13e on the BCG, 12e,
# 13d in the halo of member 2521), and an image on empty sky came out black
# (11b, 11c, 12d, 13c, all under RGB 0.01). Both are wrong in the same way -
# neither is the emitter's light.
#
# The cutouts hold the right data instead: their DATA layer is the
# continuum-subtracted narrow-band line map (for 12/13, the cube summed over
# 4962-4972 A and smoothed at sigma = 1.5 px), i.e. the Lyman-alpha emission
# itself. So these stamps are rebuilt from the line map: shape and luminance
# from the line flux, colour assigned from the line's own observed wavelength.
# That fixes the colours and the size complaint in one pass, because the same
# switch lets the footprint be cut at a lower S/N - a broadband threshold has
# to stay high to avoid pulling in the contaminating continuum, a line map
# does not. 2.0 -> 1.0 grows these footprints ~1.35-1.7x linearly (e.g. 13b
# 54 -> 150 px, 13c 38 -> 108 px) with no new blob merges.
#
# 4e rides along for a different reason: it is not an LAE, it is Source 4's
# central image, sitting 0.49" from the main deflector and drowned in its
# light, so its broadband stamp is that galaxy's gold. It gets its shape from
# the line map the same way, but its colour is inherited from its own source's
# clean images (4a-4d) rather than from a Lyman-alpha wavelength it does not
# emit at.
LAE_SOURCES = {"8", "11", "12", "13"}
LYA_REST_ANGSTROM = 1215.67
LAE_SN_THRESHOLD = 1.0  # vs ARC_SN_THRESHOLD = 2.0 for continuum stamps
LAE_ALPHA_SN_LO = 0.7  # S/N where the soft alpha ramp starts...
LAE_ALPHA_SN_HI = 3.0  # ...and where it reaches full opacity
LAE_DILATE_PX = 2  # halo around the blob that the alpha ramp may fill
LAE_FEATHER_SIGMA_PX = 1.2
LAE_ARCSINH_SOFT = 8.0  # arcsinh softening for line-flux -> luminance
LAE_LUM_FLOOR = 0.10  # keep faint outskirts coloured, not black (alpha fades)
LAE_SATURATION = 0.35  # pull the spectral colour off the gamut edge

# Saturation and the luminance floor are why the first version of these stamps
# read as neon slabs pasted over the field. Not because they were bright - the
# LAEs' 99th-percentile premultiplied luminance was 0.591, *below* the real
# arcs (0.652) and the cluster galaxies (0.835) - but because they were the
# only pure hues in a near-neutral picture. img_4c, a real arc, has mean RGB
# [0.293, 0.293, 0.302], a max/min channel ratio of 1.0; img_12a had
# [0.058, 0.385, 0.275], a ratio of 6.6, with green pinned at the top of the
# gamut. Saturation 0.85 -> 0.35 takes that ratio to 1.5. The 0.30 floor was
# the other half: it put a hard pedestal under every pixel above the S/N cut,
# so the blob had no profile, just an edge. 0.10 restores the falloff and
# costs only 8-14% of the rendered area (~4-7% linear), so the extension won
# by dropping to a line-map threshold survives.
#
# Note the two have to move together, and not in the direction you would
# guess: desaturating pushes every channel *toward white*, so it raises
# apparent luminance. At saturation 0.35 the p99 goes 0.591 -> 0.804 with no
# compensation - the fix for "too bright" would have made them brighter. Hence
# the arc-matched normalisation at the end of build_linemap_stamps.

# How the LAE colour is chosen. "lya_wavelength": the true observed
# wavelength of Lyman-alpha rendered as a spectral colour - z=3.086 -> 4967 A
# cyan-green, z=3.549 -> 5530 A yellow-green, z=4.090 -> 6188 A orange, so the
# colour is a readout of redshift.
#
# The other two modes are kept for reference but are measured dead ends. Both
# rest on the broadband hue of the source's own images, and there is no such
# thing: the LAEs have no detectable continuum in MUSE. Measured straight off
# the cube (data/cube.fits/cube.fits, three bands 4750-6100 / 6100-7600 /
# 7600-9300 A, sky lines rejected via STAT, all three Lya windows masked,
# 3-8 px annulus background, uncertainty from ~150 blank apertures of matched
# area), the stacked total S/N per source is:
#
#     source 4 (control, detected in imaging)  13.2
#     source 8                                  0.9
#     source 11                                 0.1
#     source 12                                 0.6
#     source 13                                 4.3
#
# Source 4 comes back at S/N 13, so the method works; none of the four LAEs
# does. Source 13's 4.3 is entirely image 13e (bands 1.09/2.91/3.31, ~30x
# every other image of 13), which sits on the BCG - the same contamination
# that made 13e gold in the first place. So "muse_hue" does not sample the
# emitter, it samples whatever cluster galaxy happens to lie behind it, and
# "lya_blend" samples half of that. This matches the survey paper, which has
# three of the four invisible in imaging and detected in the IFU alone.
LAE_COLOR_MODE = "lya_wavelength"
LAE_BLEND_FRAC = 0.5

# image label -> source whose *other* images its chroma is copied from
CHROMA_INHERIT = {"4e": "4"}

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
    S/N threshold - confirmed true for 3 of the 40 images)."""
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


def wavelength_to_rgb(angstrom, saturation=LAE_SATURATION):
    """A visible-spectrum wavelength as an RGB triple, normalised so the
    brightest channel is 1 (brightness is carried by the line flux, not by
    this) and desaturated toward white by `saturation`.

    Piecewise-linear hue ramp of the usual kind (Bruton). Deliberately drops
    the intensity roll-off at the ends of the visible band: it exists to model
    a human eye's falling response, and here it would just make the reddest
    emitter dimmer than the others for no reason the picture explains.
    """
    w = angstrom / 10.0  # nm
    if w < 440:
        rgb = (max(-(w - 440) / 60.0, 0.0), 0.0, 1.0)
    elif w < 490:
        rgb = (0.0, (w - 440) / 50.0, 1.0)
    elif w < 510:
        rgb = (0.0, 1.0, -(w - 510) / 20.0)
    elif w < 580:
        rgb = ((w - 510) / 70.0, 1.0, 0.0)
    elif w < 645:
        rgb = (1.0, -(w - 645) / 65.0, 0.0)
    else:
        rgb = (1.0, 0.0, 0.0)
    rgb = np.array(rgb, dtype=float)
    rgb = rgb / max(rgb.max(), 1e-9)
    return np.clip(saturation * rgb + (1.0 - saturation), 0.0, 1.0)


def _mean_chroma(stamps, labels, normalize=True):
    """Alpha-weighted mean colour of the named img_<label> stamps. With
    `normalize`, rescaled so the brightest channel is 1 - i.e. their hue with
    the brightness divided out, for use where the line flux supplies the
    brightness. Without it, the mean level is kept too, which is what an
    inherited-chroma stamp wants: 4e is a demagnified central image and should
    not come out brighter than the source's outer images. Falls back to white
    if none of the donors exist or carry light."""
    acc = np.zeros(3)
    wsum = 0.0
    for lbl in labels:
        s = stamps.get(f"img_{lbl}")
        if s is None:
            continue
        rgba = s["rgba"]
        a = rgba[..., 3]
        acc += (rgba[..., :3] * a[..., None]).sum(axis=(0, 1))
        wsum += float(a.sum())
    if wsum <= 0:
        return np.ones(3)
    mean = acc / wsum
    if mean.max() <= 1e-6:
        return np.ones(3)
    return mean / mean.max() if normalize else mean


def linemap_colors(target_labels, stamps):
    """image label -> unit-brightness RGB for every line-map stamp."""
    z_by_source = {
        source_label(lbl): z for lbl, _ra, _dec, z in LENSED_IMAGES if z is not None
    }
    siblings = {}
    for lbl, _ra, _dec, z in LENSED_IMAGES:
        siblings.setdefault(source_label(lbl), []).append(lbl)

    colors = {}
    for label in sorted(target_labels, key=_image_sort_key):
        src = source_label(label)
        if label in CHROMA_INHERIT:
            donor = CHROMA_INHERIT[label]
            others = [m for m in siblings[donor] if m != label]
            colors[label] = _mean_chroma(stamps, others, normalize=False)
            continue
        lya = wavelength_to_rgb(LYA_REST_ANGSTROM * (1.0 + z_by_source[src]))
        if LAE_COLOR_MODE == "lya_wavelength":
            colors[label] = lya
        elif LAE_COLOR_MODE == "muse_hue":
            colors[label] = _mean_chroma(stamps, siblings[src])
        elif LAE_COLOR_MODE == "lya_blend":
            broad = _mean_chroma(stamps, siblings[src])
            mix = (1.0 - LAE_BLEND_FRAC) * broad + LAE_BLEND_FRAC * lya
            colors[label] = mix / max(mix.max(), 1e-9)
        else:
            raise ValueError(f"unknown LAE_COLOR_MODE {LAE_COLOR_MODE!r}")
    return colors


def _p99_luminance(rgba):
    """99th percentile of premultiplied luminance over a stamp's lit pixels -
    how bright the thing actually reads once composited, which is the quantity
    that has to match between the line-map stamps and the broadband ones."""
    a = rgba[..., 3]
    lit = a > 0.05
    if not lit.any():
        return None
    return float(np.percentile((rgba[..., :3] * a[..., None])[lit].sum(-1) / 3.0, 99))


def build_linemap_stamps(muse_wcs, target_labels, colors, target_p99=None):
    """Same footprint segmentation as build_lensed_image_masks, but reading
    the cutout's line map for both shape and brightness instead of sampling
    muse-rgb.fits, and painting it a single assigned colour. See the
    LAE_SOURCES block above for why these images cannot use the broadband
    pixels.

    Differences from the arc pass, all following from the line map being
    background-free where the broadband image is not:
      - threshold at S/N 1.0 rather than 2.0, so the footprint is the size the
        emission really is rather than only its high-S/N core;
      - alpha is a soft ramp in S/N (0.7 -> 3.0) instead of a blurred binary
        mask, so the edge falls off the way the emission does. It is gated by
        the blob's own dilated region so a neighbouring noise peak inside the
        same bounding box cannot light up;
      - luminance is the arcsinh-stretched line flux, floored so the faint
        outskirts stay coloured while alpha takes them out.

    With `target_p99`, the finished stamps are rescaled so their median
    premultiplied p99 luminance matches it - see the note by LAE_SATURATION
    for why this cannot be folded into the saturation setting. It is one scale
    for all of them, not per stamp: the brightness differences between images
    of one source are magnification, which is the physics on display here.
    """
    out = {}
    for path, data, stat, mask, centroids, dx, dy in iter_cutouts():
        pix_centroid = {
            str(row["Source name"]): row["pix_centroid"]
            for row in centroids
            if str(row["Source name"]) in target_labels
        }
        if not pix_centroid:
            continue

        sn = np.where(mask == 1, data / np.sqrt(stat), 0.0)
        labeled, _n = ndimage.label(sn > LAE_SN_THRESHOLD)

        by_blob = {}
        for name, (cx, cy) in pix_centroid.items():
            bid = _nearest_labeled_pixel(labeled, int(round(cy)), int(round(cx)))
            if bid == 0:
                print(f"  no line-map blob for {name} ({os.path.basename(path)})")
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

            core = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
            core[ys - y0, xs - x0] = True
            region = ndimage.gaussian_filter(
                ndimage.binary_dilation(core, iterations=LAE_DILATE_PX).astype(float),
                LAE_FEATHER_SIGMA_PX,
            )
            region = np.clip(region / max(region.max(), 1e-9), 0.0, 1.0)

            sn_local = sn[y0 : y1 + 1, x0 : x1 + 1]
            ramp = np.clip(
                (sn_local - LAE_ALPHA_SN_LO) / (LAE_ALPHA_SN_HI - LAE_ALPHA_SN_LO),
                0.0,
                1.0,
            )
            alpha = region * ndimage.gaussian_filter(ramp, ARC_FEATHER_SIGMA_PX)
            alpha = np.clip(alpha / max(alpha.max(), 1e-9), 0.0, 1.0)

            flux = np.clip(data[y0 : y1 + 1, x0 : x1 + 1], 0.0, None)
            inside = flux[core]
            scale = max(float(np.percentile(inside, 99.0)), 1e-9)
            lum = np.arcsinh(flux / scale * LAE_ARCSINH_SOFT) / np.arcsinh(
                LAE_ARCSINH_SOFT
            )
            lum = LAE_LUM_FLOOR + (1.0 - LAE_LUM_FLOOR) * np.clip(lum, 0.0, 1.0)
            rgb = np.clip(lum[..., None] * colors[names[0]], 0.0, 1.0)

            mx0, mx1 = x0 + dx, x1 + dx
            my0, my1 = y0 + dy, y1 + dy
            ra, dec = muse_wcs.wcs_pix2world(
                [[(mx0 + mx1) / 2.0, (my0 + my1) / 2.0]], 0
            )[0]
            out[names[0]] = dict(
                rgba=np.dstack([rgb, alpha]),
                half_width_arcsec=(x1 - x0 + 1) / 2.0 * MUSE_PIXEL_SCALE_ARCSEC,
                half_height_arcsec=(y1 - y0 + 1) / 2.0 * MUSE_PIXEL_SCALE_ARCSEC,
                merged_labels=names,
                ra=float(ra),
                dec=float(dec),
            )

    # Match the emitters' brightness to the real arcs beside them. Only the
    # genuine Lyman-alpha stamps take part: a CHROMA_INHERIT entry (4e) is a
    # continuum image that borrowed a sibling's colour and already carries its
    # sibling's brightness level, so dragging it to the Lya level would undo
    # exactly the fix that put it there.
    if target_p99 is not None:
        lya = [k for k in out if k not in CHROMA_INHERIT]
        measured = [p for p in (_p99_luminance(out[k]["rgba"]) for k in lya) if p]
        if measured:
            scale = target_p99 / float(np.median(measured))
            for k in lya:
                rgba = out[k]["rgba"]
                rgba[..., :3] = np.clip(rgba[..., :3] * scale, 0.0, 1.0)
            print(
                f"  line-map brightness x{scale:.2f} "
                f"(p99 {np.median(measured):.3f} -> {target_p99:.3f}, "
                f"matching the broadband arcs)"
            )
    return out


def build_lensed_image_masks(muse_rgb, muse_wcs, skip_labels=frozenset()):
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
    the field still has 40 lensed images, some of which happen to touch, the
    same way an Einstein ring is still four images.

    `ra`/`dec` are the sky position of the blob's *bounding-box centre*, which
    is what the renderer must centre the stamp on. It is not the same point as
    any catalog centroid, so positioning an arc stamp on its centroid (as this
    originally did) offsets the pixels from where they belong.

    `skip_labels` are images handled by build_linemap_stamps instead. Safe to
    drop them here rather than override afterwards: checked at both
    thresholds, none of them shares a connected blob with an image that stays
    on the broadband path, so no group is left half-built."""
    out = {}
    for path, data, stat, mask, centroids, dx, dy in iter_cutouts():
        sn = np.where(mask == 1, data / np.sqrt(stat), 0.0)
        labeled, _n = ndimage.label(sn > ARC_SN_THRESHOLD)

        pix_centroid = {
            str(row["Source name"]): row["pix_centroid"]
            for row in centroids
            if str(row["Source name"]) not in skip_labels
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


def _install_footprint_stamps(stamps, masks, counts, tag):
    """Overwrite the circular img_<label> entries with segmented footprints,
    collapsing each merged group onto its first member. Returns how many
    stamps were merged away."""
    n_merged_away = 0
    for label, m in masks.items():
        key = f"img_{label}"
        if key not in stamps:
            continue  # e.g. img_8d: has a cutout but isn't in the catalog (no confirmed z)
        members = [n for n in m["merged_labels"] if f"img_{n}" in stamps]
        counts[stamps[key]["source"]] -= 1
        stamps[key] = dict(
            rgba=m["rgba"], source=tag,
            half_width_arcsec=m["half_width_arcsec"],
            half_height_arcsec=m["half_height_arcsec"],
            merged_labels=members,
            ra=m["ra"], dec=m["dec"],
        )
        counts[tag] += 1
        for n in members[1:]:
            counts[stamps.pop(f"img_{n}")["source"]] -= 1
            n_merged_away += 1
        if len(members) > 1:
            print(f"  contiguous footprint: {' + '.join(members)} -> one stamp")
    return n_merged_away


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
    #
    # Two passes over the same cutouts, splitting the images between them: the
    # Lyman-alpha emitters and 4e come from the line map (see LAE_SOURCES),
    # everything else from the broadband frame.
    line_labels = {
        r["label"] for r in lensed_rows if r["source"] in LAE_SOURCES
    } | {lbl for lbl in CHROMA_INHERIT if f"img_{lbl}" in stamps}

    n_merged_away = 0
    counts["muse_arc"] = 0
    n_merged_away += _install_footprint_stamps(
        stamps, build_lensed_image_masks(muse_rgb, muse_wcs, skip_labels=line_labels),
        counts, "muse_arc",
    )

    # colours for the line-map stamps are decided after the arc pass, because
    # 4e inherits its chroma from 4a-4d's finished arc stamps - and so is
    # their brightness, which is normalised to the level those same arcs came
    # out at rather than to a hard-coded number, so LAE_SATURATION stays a
    # knob you can turn without the emitters silently changing brightness.
    arc_p99 = [
        p
        for p in (
            _p99_luminance(s["rgba"])
            for s in stamps.values()
            if s["source"] == "muse_arc"
        )
        if p
    ]
    counts["muse_line"] = 0
    n_merged_away += _install_footprint_stamps(
        stamps,
        build_linemap_stamps(
            muse_wcs,
            line_labels,
            linemap_colors(line_labels, stamps),
            target_p99=float(np.median(arc_p99)) if arc_p99 else None,
        ),
        counts, "muse_line",
    )

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
        + [
            f"img_{lbl}"
            for lbl in ("8a", "8c", "11a", "12a", "12d", "12e", "13b", "13c",
                        "13d", "13e", "4a", "4e", "6d")
            if f"img_{lbl}" in stamps
        ]
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
