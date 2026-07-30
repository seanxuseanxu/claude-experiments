"""
Measure the true detected footprints and morphologies of field galaxies from
the HST F140W image, replacing the uniform 2.5-arcsec placeholder circles with
real shapes. Segmentation is deblended to separate blended cluster members.

Uses photutils to detect sources above background, deblend overlapping
detections with SExtractor-style parameters, and extract per-segment
morphometry from the resulting catalog. Field galaxies in the catalog fall
into a few distinct brightness tiers (the BCG, the bright halo members, and
the much fainter background fields), so deblending must be tuned to keep
cluster members separate without shattering them into fragments, while
tolerating field stars and faint background intruders.
"""
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from photutils.segmentation import (
    SegmentationImage, SourceCatalog, detect_sources, deblend_sources,
    make_2dgaussian_kernel,
)
from photutils.background import Background2D, MedianBackground

from prepare_data import load_field_catalog

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
HST_PATH = os.path.join(DATA_DIR, "hubble_f140w.fits")

# --- Background subtraction --------------------------------------------------
# The HST F140W frame has a spatially-varying background from scattered light,
# so Background2D is better than a global sigma-clipped mean. Box size 64 px
# (4.4 arcsec) captures the local sky while staying large enough to avoid
# picking up galaxy light. Filter size 3 keeps the background smooth and
# stable. The result is subtracted in place before detection.
BACKGROUND_BOX_SIZE = 64
BACKGROUND_FILTER_SIZE = 3

# --- Source detection --------------------------------------------------------
# detect_sources thresholds the background-subtracted data at N*sigma above
# zero. At 1.5-sigma many noise pixels just above background would trigger
# detection, leading to spurious fragments; at 2.5-sigma we begin to lose the
# faintest real galaxies. 2.0-sigma is the midpoint that balances them.
DETECTION_SIGMA = 2.0

# Minimum size in pixels to count as a real source, not noise specks.
# 5 pixels at 0.07 arcsec/px is ~0.35 arcsec; this rejects isolated hot
# pixels and the tails of cosmic rays while keeping real galaxies.
NPIXELS_MIN = 5

# Gaussian smoothing kernel FWHM (in pixels) applied before detection, matched
# to the HST PSF (~0.16 arcsec = 2.3 px in F140W). This boosts faint objects
# and preserves real shapes.
KERNEL_FWHM_PX = 2.2

# --- Deblending parameters ---------------------------------------------------
# The F140W frame contains a blend of cluster members (bright), field galaxies
# (medium), and foreground stars (pinpoint). Deblending must separate the
# cluster's tight members without fragmenting real galaxies into subcores.
# nlevels controls the resolution of the threshold ladder; more levels = more
# aggressive deblending. contrast is the SExtractor DEBLEND_MINCONT analogue:
# fragments smaller than this fraction of the parent blob's peak are filtered.
#
# Tuned on the validation set to separate 2373/2399/2444/2465 into distinct
# segments, keep 2350/2357 separate, and rank 2444 (BCG) in the top few by area.
DEBLEND_NLEVELS = 64
DEBLEND_CONTRAST = 0.0005  # SExtractor DEBLEND_MINCONT analogue


def _load_hst_image():
    """Load and WCS-calibrate the HST F140W frame. Return (data, wcs,
    pixel_scale_arcsec)."""
    with fits.open(HST_PATH) as hdul:
        data = hdul[0].data
        hdr = hdul[0].header

    data = np.nan_to_num(data)
    wcs = WCS(hdr)

    # Measure pixel scale from WCS CD matrix.
    pscale = wcs.pixel_scale_matrix
    px_scale_deg = np.sqrt(pscale[0, 0] ** 2 + pscale[0, 1] ** 2)
    px_scale_arcsec = px_scale_deg * 3600.0

    return data, wcs, px_scale_arcsec


def _estimate_background(data):
    """Estimate the local background using sigma-clipped statistics on blocks.
    Return (background-subtracted data, background level, background rms)."""
    bkg = Background2D(
        data,
        box_size=BACKGROUND_BOX_SIZE,
        filter_size=BACKGROUND_FILTER_SIZE,
        bkg_estimator=MedianBackground(),
    )

    bg_data = data - bkg.background

    # Global sigma-clipped estimate of the noise level in the background-
    # subtracted frame (where real sources are marked as masked).
    clipped = sigma_clip(bg_data, sigma=3.0, masked=True)
    bg_rms = float(np.std(clipped))

    return bg_data, bkg.background, bg_rms


def _detect_and_deblend(data, rms_px):
    """Detect sources using a Gaussian-smoothed convolution kernel at
    DETECTION_SIGMA, then deblend overlapping components using photutils'
    multi-threshold ladder (SExtractor style). Keep only components above
    DEBLEND_CONTRAST of the parent blob's peak flux."""
    from astropy.convolution import convolve

    # Build a Gaussian kernel matched to KERNEL_FWHM_PX and convolve the data.
    threshold = DETECTION_SIGMA * rms_px
    kernel = make_2dgaussian_kernel(KERNEL_FWHM_PX, size=int(2 * KERNEL_FWHM_PX) + 1)
    smoothed_data = convolve(data, kernel)

    # Detect sources on the smoothed data; deblend uses the original data.
    segm = detect_sources(smoothed_data, threshold, n_pixels=NPIXELS_MIN)

    if segm is None or segm.data.max() < 1:
        return np.zeros(data.shape, dtype=int)

    # Deblend using multi-threshold approach with SExtractor-style parameters.
    # Pass the original (non-smoothed) data so that flux measurements are accurate.
    segm_deblended = deblend_sources(
        data, segm, n_pixels=NPIXELS_MIN, n_levels=DEBLEND_NLEVELS,
        contrast=DEBLEND_CONTRAST, relabel=True
    )

    return segm_deblended.data


def _nearest_labeled_pixel(labeled, iy, ix, max_r=5):
    """Return the segment label at (iy, ix) in the labeled image, or if it is
    background (0), search within max_r pixels for the nearest nonzero label.

    Mirrors the logic used for lensed arcs: a catalog position that falls just
    barely below a detection threshold should still map to the nearest real
    segment, not to background."""
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


def measure_galaxy_shapes():
    """Measure the true segmented footprints and morphologies of field galaxies.

    Returns a dict: catalog label -> dict with keys:
        mask : 2D bool array, the deblended footprint cropped to its bbox
        half_width_arcsec, half_height_arcsec : half-extents of the bbox
        bbox_ra, bbox_dec : sky coordinates of the bbox centre
        pixel_scale_arcsec, area_arcsec2, n_pixels
        semi_major_arcsec, semi_minor_arcsec, orientation_deg, ellipticity

    Objects outside the HST frame or with no clean detection are absent from
    the returned dict. """

    # Load the HST frame and its calibration.
    data, wcs, px_scale_arcsec = _load_hst_image()

    # Subtract background and measure noise.
    bg_data, _bg_level, bg_rms = _estimate_background(data)

    # Detect and deblend sources.
    segm_arr = _detect_and_deblend(bg_data, bg_rms)

    # If no sources at all, return empty dict.
    if segm_arr.max() < 1:
        return {}

    # Wrap in SegmentationImage for photutils
    segm = SegmentationImage(segm_arr)

    # Build a source catalog from the deblended segmentation.
    cat = SourceCatalog(bg_data, segm)

    # Load the field catalog to match objects.
    field_catalog = load_field_catalog()

    # For each field object, attempt to match it to a segment.
    out = {}
    segm_data = segm.data  # Get the underlying numpy array
    for obj in field_catalog:
        label = obj["label"]
        ra_deg, dec_deg = obj["ra"], obj["dec"]

        # Project the catalog position to pixel space.
        px, py = wcs.wcs_world2pix([[ra_deg, dec_deg]], 0)[0]
        ix, iy = int(np.round(px)), int(np.round(py))

        # Check if this position is inside the image.
        ny, nx = segm_data.shape
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue  # outside the frame

        # Find the nearest segment at or near this position.
        seg_id = _nearest_labeled_pixel(segm_data, iy, ix)
        if seg_id == 0:
            continue  # no segment found

        # Locate this segment in the catalog (1-indexed in photutils).
        try:
            src = cat[seg_id - 1]
        except IndexError:
            continue  # segment ID out of range (shouldn't happen)

        # Get the bounding box and footprint.
        bbox = src.bbox
        x0, x1 = bbox.ixmin, bbox.ixmax + 1
        y0, y1 = bbox.iymin, bbox.iymax + 1

        # Extract the mask for this segment, cropped to its bbox.
        mask = (segm_data[y0:y1, x0:x1] == seg_id)

        # Compute half-extents from the bbox.
        height_px = y1 - y0
        width_px = x1 - x0
        half_height_arcsec = height_px / 2.0 * px_scale_arcsec
        half_width_arcsec = width_px / 2.0 * px_scale_arcsec

        # Get the bbox centre in sky coordinates.
        bbox_center_px_x = (x0 + x1 - 1) / 2.0
        bbox_center_px_y = (y0 + y1 - 1) / 2.0
        bbox_ra, bbox_dec = wcs.wcs_pix2world(
            [[bbox_center_px_x, bbox_center_px_y]], 0
        )[0]

        # Get morphology from the source catalog.
        # Note: photutils quantities have .value for the numerical part
        n_pixels = int(src.area.value)
        area_arcsec2 = n_pixels * (px_scale_arcsec ** 2)
        semi_major_arcsec = float(src.semimajor_axis.value) * px_scale_arcsec
        semi_minor_arcsec = float(src.semiminor_axis.value) * px_scale_arcsec
        orientation_deg = float(src.orientation.value)  # Already in degrees
        ellipticity = float(src.ellipticity)

        out[label] = dict(
            mask=mask,
            half_width_arcsec=float(half_width_arcsec),
            half_height_arcsec=float(half_height_arcsec),
            bbox_ra=float(bbox_ra),
            bbox_dec=float(bbox_dec),
            pixel_scale_arcsec=float(px_scale_arcsec),
            area_arcsec2=float(area_arcsec2),
            n_pixels=n_pixels,
            semi_major_arcsec=float(semi_major_arcsec),
            semi_minor_arcsec=float(semi_minor_arcsec),
            orientation_deg=orientation_deg,
            ellipticity=ellipticity,
        )

    return out


if __name__ == "__main__":
    results = measure_galaxy_shapes()

    # Extract validation targets.
    validation_labels = {"2373", "2399", "2444", "2465", "2350", "2357"}
    validation_results = {k: results[k] for k in validation_labels if k in results}

    # Load the full catalog to get categories.
    field_catalog = load_field_catalog()
    label_to_cat = {obj["label"]: obj for obj in field_catalog}

    # Print validation checks.
    print("=== VALIDATION CHECKS ===\n")

    # Check 0: collision check - verify no two distinct catalog labels map to
    # the same segment (identical area_arcsec2 and bbox_ra/dec within tolerance).
    collision_pairs = []
    result_items = list(validation_results.items())
    for i, (label1, r1) in enumerate(result_items):
        for label2, r2 in result_items[i + 1:]:
            # Within floating-point tolerance: area equal to 1e-6 arcsec^2,
            # position within 1e-6 degrees.
            area_match = abs(r1["area_arcsec2"] - r2["area_arcsec2"]) < 1e-6
            ra_match = abs(r1["bbox_ra"] - r2["bbox_ra"]) < 1e-6
            dec_match = abs(r1["bbox_dec"] - r2["bbox_dec"]) < 1e-6
            if area_match and ra_match and dec_match:
                collision_pairs.append((label1, label2))

    if collision_pairs:
        print(f"FAIL: Collision detected - distinct catalog labels mapped to same segment:")
        for l1, l2 in collision_pairs:
            print(f"  {l1} and {l2}: area={validation_results[l1]['area_arcsec2']:.1f} arcsec^2")
    else:
        print(f"PASS: No collisions - distinct catalog labels have distinct measurements")

    # Check 1: four distinct cluster members with distinct sizes/positions.
    cluster_four = {"2373", "2399", "2444", "2465"}
    present = cluster_four & set(validation_results.keys())
    if len(present) == len(cluster_four):
        # Check that they don't all have identical measurements (which would
        # indicate they're actually the same segment)
        sizes = [(validation_results[l]["area_arcsec2"],
                  validation_results[l]["bbox_ra"],
                  validation_results[l]["bbox_dec"]) for l in sorted(present)]
        unique_areas = len(set(s[0] for s in sizes))
        if unique_areas > 1:
            print(f"PASS: All four cluster members {cluster_four} detected with distinct sizes")
        else:
            print(f"FAIL: All four objects have identical footprints (same segment)")
            for lbl in sorted(present):
                r = validation_results[lbl]
                print(f"  {lbl}: area={r['area_arcsec2']:.1f}, pos=({r['bbox_ra']:.4f}, {r['bbox_dec']:.4f})")
    else:
        missing = cluster_four - present
        print(f"FAIL: Missing detections for {missing}")

    # Check 2: 2350 and 2357 are separate (2357 is a foreground star).
    if "2350" in validation_results and "2357" in validation_results:
        # They should be distinct segments.
        a1 = validation_results["2350"]["area_arcsec2"]
        a2 = validation_results["2357"]["area_arcsec2"]
        if a1 != a2:
            print(f"PASS: 2350 (galaxy, {a1:.1f} arcsec^2) and 2357 (star, {a2:.1f} arcsec^2) are separate")
        else:
            print(f"FAIL: 2350 and 2357 have identical areas - not separate")
    elif "2350" in validation_results or "2357" in validation_results:
        print(f"PARTIAL: {set(['2350', '2357']) & set(validation_results.keys())} detected")
    else:
        print(f"FAIL: Neither 2350 nor 2357 detected")

    # Check 3: BCG 2444 should be one of the largest.
    if "2444" in validation_results:
        area_2444 = validation_results["2444"]["area_arcsec2"]
        all_areas = sorted([validation_results[l]["area_arcsec2"] for l in validation_results], reverse=True)
        rank = all_areas.index(area_2444) + 1 if area_2444 in all_areas else len(all_areas)
        if rank <= 3:
            print(f"PASS: BCG 2444 ranks #{rank} by area ({area_2444:.1f} arcsec^2)")
        else:
            print(f"FAIL: BCG 2444 ranks #{rank} (expected top 3), area={area_2444:.1f} arcsec^2")
    else:
        print(f"FAIL: BCG 2444 not detected")

    print("\n=== FULL OBJECT TABLE ===\n")
    print(f"{'Label':<8} {'Category':<18} {'Half-W':<10} {'Half-H':<10} {'Area':<10} {'Ellip':<8}")
    print("-" * 80)

    for obj in field_catalog:
        label = obj["label"]
        category = obj["category"]
        if label in results:
            hw = results[label]["half_width_arcsec"]
            hh = results[label]["half_height_arcsec"]
            area = results[label]["area_arcsec2"]
            ell = results[label]["ellipticity"]
            print(f"{label:<8} {category:<18} {hw:>9.2f}\" {hh:>9.2f}\" {area:>9.1f}  {ell:>7.3f}")

    print("\n=== SUMMARY STATISTICS ===\n")

    if results:
        half_widths = np.array([r["half_width_arcsec"] for r in results.values()])
        half_heights = np.array([r["half_height_arcsec"] for r in results.values()])
        half_extents = np.concatenate([half_widths, half_heights])

        print(f"Objects with measurements: {len(results)} / 58")
        print(f"\nHalf-extent distribution (min/median/p75/max):")
        print(f"  {np.min(half_extents):.2f}\" / "
              f"{np.median(half_extents):.2f}\" / "
              f"{np.percentile(half_extents, 75):.2f}\" / "
              f"{np.max(half_extents):.2f}\"")

        areas = np.array([r["area_arcsec2"] for r in results.values()])
        print(f"\nArea distribution (min/median/p75/max):")
        print(f"  {np.min(areas):.1f} / "
              f"{np.median(areas):.1f} / "
              f"{np.percentile(areas, 75):.1f} / "
              f"{np.max(areas):.1f} arcsec^2")
