"""
Check that the cluster billboard and the individual stamps land on the same
sky, so the cross-dissolve is a dissolve and not a slide.

This exists because the two are positioned by completely different routes.
The billboard is one image pinned at the cluster plane, centred on the MUSE
frame's own centre; each stamp is placed by converting its RA/Dec to a
transverse comoving offset from the scene origin. The scene origin is the main
deflector, which is 10.9" from the middle of the MUSE frame - so the moment
the origin moved off the frame centre, the billboard had to be offset by
exactly that much or every galaxy would have slid out from under its own
photo. Nothing in the renderer would have complained; it would just have
looked slightly wrong.

Two checks:

1. Numeric, and the one that actually proves it. For a set of sky positions,
   compare where the billboard puts that piece of sky against where the object
   machinery puts an object at the same coordinates. Evaluated at the cluster
   redshift, where the two must agree exactly - a residual here is a pure
   registration error, with no parallax mixed in. Reported in arcsec.

2. Visual, output/registration_zoom.png: the same frame rendered billboard-only
   and stamps-only, plus an overlay of one on the other, zoomed on the
   deflector. A slip too small to matter numerically is still obvious here.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import flythrough as fly
from cutout_data import muse_wcs_2d
from prepare_data import (
    CLUSTER_Z,
    DEFLECTOR_ID,
    FIELD_CENTER_DEC,
    FIELD_CENTER_RA,
    load_field_catalog,
    sky_to_transverse_mpc,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_PATH = os.path.join(ROOT, "output/registration_zoom.png")

# Mid-dissolve: billboard and stamps both at ~half opacity, which is where a
# mismatch would be most visible in the finished animation.
CHECK_DEPTH_MPC = fly.DISSOLVE_END_DEPTH + 0.5 * (
    fly.DISSOLVE_START_DEPTH - fly.DISSOLVE_END_DEPTH
)
CHECK_CAM_Z = fly.CLUSTER_D_C - CHECK_DEPTH_MPC
ZOOM_HALF = 0.30  # view-space half-extent of the zoomed panels
PANEL_PX = 640
TOLERANCE_ARCSEC = 0.02  # bulk offset budget: a tenth of a MUSE pixel
MAX_CORNER_ARCSEC = 0.10  # single-point budget, incl. the tangent-plane term


def _billboard_view_position(ra_deg, dec_deg, cam_z):
    """Where the billboard draws the piece of sky at (ra, dec), in view space.

    Mirrors the imshow() call in render_frame: the image spans `extent`
    edge-to-edge with origin='lower', so pixel centre i of nx sits at
    fraction (i + 0.5) / nx across it.
    """
    f = 1.0 / np.tan(np.radians(fly.FOV_DEG) / 2.0)
    bb_depth = fly.CLUSTER_D_C - cam_z
    half_w = np.radians(fly.MUSE_W_ARCSEC / 3600.0 / 2.0) * fly.CLUSTER_D_C
    half_h = np.radians(fly.MUSE_H_ARCSEC / 3600.0 / 2.0) * fly.CLUSTER_D_C
    bx, by = f * half_w / bb_depth, f * half_h / bb_depth
    bcx, bcy = f * fly.BILLBOARD_X / bb_depth, f * fly.BILLBOARD_Y / bb_depth

    wcs = muse_wcs_2d()
    with fits.open(os.path.join(ROOT, "data/muse-rgb.fits")) as hdul:
        nx = int(hdul[0].header["NAXIS1"])
        ny = int(hdul[0].header["NAXIS2"])
    ipx, ipy = wcs.wcs_world2pix([[ra_deg, dec_deg]], 0)[0]
    return (
        (bcx - bx) + (ipx + 0.5) / nx * 2 * bx,
        (bcy - by) + (ipy + 0.5) / ny * 2 * by,
    )


def _object_view_position(ra_deg, dec_deg, cam_z):
    """Where an object at (ra, dec) sitting in the cluster plane is drawn."""
    f = 1.0 / np.tan(np.radians(fly.FOV_DEG) / 2.0)
    x, y, d_c = sky_to_transverse_mpc(ra_deg, dec_deg, CLUSTER_Z)
    depth = d_c - cam_z
    return f * x / depth, f * y / depth


def numeric_check(cam_z=CHECK_CAM_Z):
    """Residual between the two placements, in arcsec on the sky, at the
    deflector, at every cluster member inside the MUSE footprint, and at the
    four frame corners.

    Reported two ways, because they mean different things:

    - the *mean residual vector* is a bulk shift of the whole photo against
      the whole catalog. That is the registration bug this file exists to
      catch, and it should be zero.
    - the *max magnitude* also contains an irreducible term from
      sky_to_transverse_mpc's flat tangent-plane approximation, which the
      billboard's proper TAN WCS does not share. It is zero at the field
      centre and grows as roughly r^2, so it shows up only at the frame
      corners and is bounded by the frame's own size.
    """
    arcsec_per_view_unit = (
        fly.FOV_DEG * 3600.0 / 2.0
    )  # view space is +-1 across the vertical FOV

    wcs = muse_wcs_2d()
    with fits.open(os.path.join(ROOT, "data/muse-rgb.fits")) as hdul:
        nx = int(hdul[0].header["NAXIS1"])
        ny = int(hdul[0].header["NAXIS2"])

    def inside(ra, dec):
        ipx, ipy = wcs.wcs_world2pix([[ra, dec]], 0)[0]
        return 0 <= ipx <= nx - 1 and 0 <= ipy <= ny - 1

    # Only positions the billboard actually covers: an object outside the MUSE
    # footprint has no billboard pixels to be registered against (its stamp
    # comes from HST or the placeholder), so comparing there measures nothing.
    points = [("deflector " + DEFLECTOR_ID, FIELD_CENTER_RA, FIELD_CENTER_DEC)]
    n_outside = 0
    for r in load_field_catalog():
        if r["category"] != "cluster_member":
            continue
        if inside(r["ra"], r["dec"]):
            points.append((r["label"], r["ra"], r["dec"]))
        else:
            n_outside += 1
    for name, cx, cy in (
        ("corner BL", 0, 0),
        ("corner BR", nx - 1, 0),
        ("corner TL", 0, ny - 1),
        ("corner TR", nx - 1, ny - 1),
    ):
        ra, dec = wcs.wcs_pix2world([[cx, cy]], 0)[0]
        points.append((name, float(ra), float(dec)))

    d = np.array(
        [
            np.subtract(
                _billboard_view_position(ra, dec, cam_z),
                _object_view_position(ra, dec, cam_z),
            )
            for _n, ra, dec in points
        ]
    ) * arcsec_per_view_unit
    mag = np.hypot(d[:, 0], d[:, 1])
    shift = np.hypot(*d.mean(axis=0))
    worst = int(np.argmax(mag))

    print(
        f"Registration residual over {len(points)} sky positions inside the "
        f"MUSE footprint ({n_outside} cluster members outside it skipped):"
    )
    print(f"  bulk shift  {shift:.4f}\"  (tolerance {TOLERANCE_ARCSEC}\")")
    print(f"  median      {np.median(mag):.4f}\"")
    print(f"  max         {mag[worst]:.4f}\"  at {points[worst][0]}")
    if shift > TOLERANCE_ARCSEC:
        raise AssertionError(
            f"billboard and stamps are offset as a whole by {shift:.3f}\", "
            f"tolerance {TOLERANCE_ARCSEC}\" - check BILLBOARD_X/Y in "
            f"flythrough.py against FIELD_CENTER_RA/DEC in prepare_data.py"
        )
    if mag[worst] > MAX_CORNER_ARCSEC:
        raise AssertionError(
            f"worst-case residual {mag[worst]:.3f}\" at {points[worst][0]} "
            f"exceeds {MAX_CORNER_ARCSEC}\"; too large to be the tangent-plane "
            f"term alone"
        )
    print(f"  OK - no bulk offset, worst case under {MAX_CORNER_ARCSEC}\" "
          f"({MAX_CORNER_ARCSEC / 0.2:.1f} MUSE pixels) at the frame corners")
    return mag


def _render_raster(objects, starfield, billboard_img, z_lookup, forced_bb_alpha):
    """One zoomed frame with the billboard forced fully on or fully off,
    returned as an RGB array. Goes through the real render_frame so the panels
    cannot drift from what the animation actually draws."""
    original = fly.billboard_alpha
    fly.billboard_alpha = lambda cam_z: forced_bb_alpha
    try:
        # facecolor matters: render_frame calls ax.axis("off"), which hides the
        # axes patch, so the *figure* background is what shows behind the sky
        fig = plt.figure(
            figsize=(PANEL_PX / 100.0, PANEL_PX / 100.0), dpi=100, facecolor="black"
        )
        ax = fig.add_axes([0, 0, 1, 1])
        fly.render_frame(
            ax, CHECK_CAM_Z, objects, starfield, billboard_img, z_lookup,
            show_labels=False, show_hud=False,
        )
        ax.set_xlim(-ZOOM_HALF, ZOOM_HALF)
        ax.set_ylim(-ZOOM_HALF, ZOOM_HALF)
        fig.canvas.draw()
        raster = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255.0
        plt.close(fig)
        return raster
    finally:
        fly.billboard_alpha = original


def visual_check():
    field_rows, lensed_rows, stamps = fly.load_scene()
    objects = fly.build_objects(field_rows, lensed_rows, stamps)
    starfield = fly.build_starfield()
    billboard_img = plt.imread(os.path.join(ROOT, "output/muse_rgb.png"))
    z_lookup = fly.build_redshift_lookup()

    bb = _render_raster(objects, starfield, billboard_img, z_lookup, 1.0)
    st = _render_raster(objects, starfield, billboard_img, z_lookup, 0.0)

    # overlay: billboard as grey, stamps as magenta on top of it. If the two
    # are registered the magenta sits on the grey blobs; if not, it sits
    # beside them.
    grey = bb.mean(axis=-1)
    st_lum = st.mean(axis=-1)
    w = np.clip(st_lum / max(st_lum.max(), 1e-6) * 2.5, 0, 1)[..., None]
    magenta = np.array([1.0, 0.15, 0.85])
    over = np.clip(np.dstack([grey * 0.75] * 3) * (1 - w) + magenta * w, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="black")
    for ax, img, title in zip(
        axes,
        (bb, st, over),
        (
            "billboard only (MUSE frame)",
            "stamps only (per-object cutouts)",
            "overlay: billboard grey, stamps magenta",
        ),
    ):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"Registration check - zoomed on deflector {DEFLECTOR_ID}, "
        f"cam {CHECK_DEPTH_MPC:.0f} Mpc in front of the cluster\n"
        "magenta should sit on grey; the magenta with no grey under it is the "
        "Lyman-alpha emitters, which are absent from the broadband billboard "
        "by construction",
        color="white",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=110, facecolor="black")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    numeric_check()
    visual_check()
