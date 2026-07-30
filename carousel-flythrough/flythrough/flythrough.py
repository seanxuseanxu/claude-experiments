"""
Render the Carousel flythrough animation: a camera dollies along the line of
sight from near z=0 out past z~4.2, through the real 3D-positioned field
galaxies, cluster members, and lensed source images built by prepare_data.py
and prepare_imagery.py.

Usage:
    python3 flythrough.py --preview      # a handful of low-res still frames
    python3 flythrough.py                # full-length mp4
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from prepare_data import (
    CLUSTER_Z,
    COSMO,
    MUSE_CENTER_DEC,
    MUSE_CENTER_RA,
    MUSE_HALF_H_ARCSEC,
    MUSE_HALF_W_ARCSEC,
    load_field_catalog,
    load_lensed_sources,
    sky_to_transverse_mpc,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "output")

SHOW_LABELS = True
SHOW_HUD = True

# --- Camera / flight parameters -------------------------------------------
# The real field is only a few arcmin across (MUSE footprint ~1.1x1.2
# arcmin), so a normal-photo FOV would shrink everything to sub-pixel
# specks. Use a narrow, telephoto-like FOV sized to the data's own angular
# scale instead.
FOV_DEG = 8.0 / 60.0  # vertical field of view (8 arcmin)
Z_START = 40.0  # Mpc, just in front of the nearest real object (star at ~0.26 Mpc excluded)
Z_END = 7550.0  # Mpc, just past z=4.09
FPS = 30
DURATION_S = 26.0
N_FRAMES = int(FPS * DURATION_S)

# Cluster billboard true footprint (340x348 px @ 0.2"/px), measured off the
# frame's own WCS.
MUSE_W_ARCSEC = 2 * MUSE_HALF_W_ARCSEC
MUSE_H_ARCSEC = 2 * MUSE_HALF_H_ARCSEC
CLUSTER_D_C = COSMO.comoving_distance(CLUSTER_Z).value

# Where the billboard's own centre sits in scene coordinates. The scene origin
# is the main deflector (see DEFLECTOR_ID in prepare_data.py), which is 10.9"
# from the middle of the MUSE frame, so the photo is *not* centred on the
# origin. Drawing it there anyway would slide it ~21% of its own half-width
# off the galaxies it contains, and the cross-dissolve would visibly slip.
BILLBOARD_X, BILLBOARD_Y, _ = sky_to_transverse_mpc(
    MUSE_CENTER_RA, MUSE_CENTER_DEC, CLUSTER_Z
)

STAR_COLOR = (1.0, 1.0, 0.95)

# --- billboard -> galaxies cross-dissolve ---------------------------------
# From far away the field should just read as the MUSE picture. As the camera
# closes in, that picture dissolves and the individual galaxies it is made of
# take over. Driven by how far the camera still is from the cluster plane, and
# finished well before the camera reaches it, so we never fly into a flat card.
#
# Only objects at or behind the cluster plane cross-dissolve. The three
# genuine foreground objects (FGD at 367 Mpc, 2046 at 1104, 2292 at 1519) are
# in front of the billboard and always drawn: the camera passes FGD long
# before the dissolve even starts, so gating it on the dissolve means it is
# never seen at all. Objects outside the MUSE footprint (HST fallback or
# placeholder) aren't in the billboard picture either, so they always draw too.
DISSOLVE_START_DEPTH = 1400.0  # Mpc in front of the cluster: billboard solid
DISSOLVE_END_DEPTH = 600.0  # billboard fully replaced by individual galaxies
FOREGROUND_MARGIN_MPC = 200.0  # how far in front of the cluster still counts
# as foreground (nearest cluster member is at 1840, next object in is at 1519)


def ease_in_out(t):
    """Smoothstep easing, t in [0,1] -> [0,1]."""
    return t * t * (3 - 2 * t)


def camera_z_of_frame(i, n_frames):
    """Comoving distance of the camera at frame i, with ease-in/out at the
    start and end of the flight (linear cruise in between)."""
    t = i / max(n_frames - 1, 1)
    ease_frac = 0.12
    if t < ease_frac:
        te = ease_in_out(t / ease_frac) * ease_frac
    elif t > 1 - ease_frac:
        tail = (t - (1 - ease_frac)) / ease_frac
        te = (1 - ease_frac) + ease_in_out(tail) * ease_frac
    else:
        te = t
    return Z_START + te * (Z_END - Z_START)


def load_scene():
    field_rows = load_field_catalog()
    lensed_rows = load_lensed_sources()
    stamps = np.load(os.path.join(OUT_DIR, "stamps.npy"), allow_pickle=True).item()
    return field_rows, lensed_rows, stamps


def build_starfield(n=250, seed=42, d_c=9000.0, spread=6.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-spread, spread, n)
    y = rng.uniform(-spread, spread, n)
    d = rng.uniform(d_c * 0.8, d_c * 1.2, n)
    b = rng.uniform(0.3, 1.0, n)
    return x, y, d, b


def project(x, y, d_c, cam_z, fov_deg=FOV_DEG, aspect=1.0):
    """Perspective-project a point at transverse offset (x,y) Mpc and
    comoving distance d_c Mpc, given the camera's comoving position cam_z
    along the line of sight. Returns (px, py, depth) in normalized
    [-1, 1]-ish view-space, where depth = d_c - cam_z (Mpc in front of cam)."""
    depth = d_c - cam_z
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    px = f * x / depth
    py = f * y / depth * aspect
    return px, py, depth


def build_redshift_lookup():
    """d_c (Mpc) -> z, via interpolation over a dense precomputed grid
    (avoids repeated astropy cosmology inversion per frame)."""
    zs = np.linspace(0.0, 4.5, 2000)
    dcs = COSMO.comoving_distance(zs).value
    return lambda d: np.interp(d, dcs, zs)


def object_angular_halfsize_deg(d_c, radius_arcsec):
    """Convert a stamp's real angular radius (arcsec, fixed at capture time)
    back to a physical transverse half-size in Mpc at its own comoving
    distance, matching how prepare_imagery.py cut it out."""
    return np.radians(radius_arcsec / 3600.0) * d_c


class Object3D:
    __slots__ = (
        "x", "y", "d_c", "z", "rgba", "half_w_mpc", "half_h_mpc", "kind",
        "label", "source", "dissolve",
    )

    def __init__(self, x, y, d_c, z, rgba, half_w_mpc, half_h_mpc, kind, label,
                 source=None, dissolve=False):
        self.x = x
        self.y = y
        self.d_c = d_c
        self.z = z
        self.rgba = rgba
        self.half_w_mpc = half_w_mpc
        self.half_h_mpc = half_h_mpc
        self.kind = kind  # 'field', 'lensed'
        self.label = label
        self.source = source
        # True if this object's light is already in the cluster billboard, so
        # it should fade in as the billboard fades out instead of being drawn
        # on top of its own photo. See the cross-dissolve note above.
        self.dissolve = dissolve


def _in_billboard(stamp, d_c):
    """Is this object's light part of the MUSE billboard picture? Only if it
    was cut from the MUSE frame at all, and only if it sits at or behind the
    cluster plane the billboard is pinned to.

    'muse_line' counts even though, strictly, it does not: those stamps are
    narrow-band Lyman-alpha, which is exactly the light the broadband
    billboard does *not* contain. Drawing them on top of it from frame one
    would be honest and would also look like a separate overlay pasted over
    the photo. Dissolving them in with everything else keeps one visual rule -
    the picture resolves into its objects - and the emitters simply arrive as
    the flat frame lets go."""
    return (
        stamp["source"] in ("muse", "muse_arc", "muse_line")
        and d_c >= CLUSTER_D_C - FOREGROUND_MARGIN_MPC
    )


def build_objects(field_rows, lensed_rows, stamps):
    objects = []
    for r in field_rows:
        if r["category"] == "star":
            continue  # excluded: this is a foreground Milky Way star, not a galaxy
        s = stamps.get(r["label"])
        if s is None:
            continue
        half_w = object_angular_halfsize_deg(r["d_c"], s["half_width_arcsec"])
        half_h = object_angular_halfsize_deg(r["d_c"], s["half_height_arcsec"])
        objects.append(
            Object3D(
                r["x"], r["y"], r["d_c"], r["z"], s["rgba"], half_w, half_h,
                "field", r["label"], dissolve=_in_billboard(s, r["d_c"]),
            )
        )

    # Images that share a connected footprint are cached under the first of
    # them (prepare_imagery drops the rest), so their catalog rows have no
    # stamp and are skipped here. They are still 40 images in the field -
    # some of them just touch, and get drawn as one piece of sky.
    for r in lensed_rows:
        s = stamps.get(f"img_{r['label']}")
        if s is None:
            continue
        # An arc stamp is centred on its footprint's bounding box, which is
        # not the catalog centroid - take the position from the stamp so the
        # pixels land where they actually are.
        if "ra" in s:
            x, y, _ = sky_to_transverse_mpc(s["ra"], s["dec"], r["z"])
        else:
            x, y = r["x"], r["y"]
        half_w = object_angular_halfsize_deg(r["d_c"], s["half_width_arcsec"])
        half_h = object_angular_halfsize_deg(r["d_c"], s["half_height_arcsec"])
        objects.append(
            Object3D(
                x, y, r["d_c"], r["z"], s["rgba"], half_w, half_h,
                "lensed", r["label"], r["source"],
                dissolve=_in_billboard(s, r["d_c"]),
            )
        )
    return objects


def billboard_alpha(cam_z):
    """1 while the field should read as the flat MUSE picture, 0 once it has
    been fully replaced by the individual galaxies, smoothstepped between."""
    bb_depth = CLUSTER_D_C - cam_z
    span = DISSOLVE_START_DEPTH - DISSOLVE_END_DEPTH
    t = np.clip((bb_depth - DISSOLVE_END_DEPTH) / span, 0.0, 1.0)
    return ease_in_out(t)


NEAR_CLIP = 3.0  # Mpc; objects closer than this to the camera are culled
VIEW_HALF = 1.0  # normalized view-space half-extent (matches xlim/ylim)
# How far ahead of an object its label starts fading up. This is what sets how
# long a label is legible, and 500 was much too short: at ~11 Mpc/frame the
# ramp only reached full strength 75 Mpc out, giving a median 1.15 s per
# source, and sources 9 and 11 - both far enough off-axis to leave the frame
# edge early - got exactly one frame each, at peak alpha 0.17. 1400 gives a
# median 3.87 s and brings 9 and 11 up to 2.8 / 3.0 s at peak alpha 0.82.
#
# The cost is crowding, which is why this is not larger: 1400 puts at most 6
# labels on screen at once (mean 3.2), and at fontsize 11 no pair of label
# boxes overlaps in any of the 780 frames. 2000 was tried and reaches 8.
LABEL_FADE_IN_MPC = 1400.0


def _alpha_for_depth_fade(depth, fade_in):
    """Label opacity from the object's depth in front of the camera: 0 beyond
    fade_in, ramping 0 -> 1 from fade_in down to fade_in * 0.15, then full
    until the near clip.

    There is deliberately no far-side fade here. An object never recedes in
    this flight - the camera only moves forward - so the way a label ends is
    the object swelling past the size cap, and that is handled by close_alpha
    at the call site, which multiplies this."""
    if depth > fade_in:
        return 0.0
    if depth > fade_in * 0.15:
        return np.clip(1.0 - (depth - fade_in * 0.15) / (fade_in * 0.85), 0, 1)
    if depth > NEAR_CLIP:
        return 1.0
    return 0.0


def render_frame(ax, cam_z, objects, starfield, billboard_img, z_lookup, show_labels, show_hud):
    ax.clear()
    ax.set_facecolor("black")
    ax.set_xlim(-VIEW_HALF, VIEW_HALF)
    ax.set_ylim(-VIEW_HALF, VIEW_HALF)
    ax.set_aspect("equal")
    ax.axis("off")

    f = 1.0 / np.tan(np.radians(FOV_DEG) / 2.0)

    # --- starfield (background, painted first) ---
    sx, sy, sd, sb = starfield
    depth = sd - cam_z
    visible = depth > NEAR_CLIP
    px = f * sx[visible] / depth[visible]
    py = f * sy[visible] / depth[visible]
    onscreen = (np.abs(px) < VIEW_HALF * 1.2) & (np.abs(py) < VIEW_HALF * 1.2)
    sizes = np.clip(6.0 * (2000.0 / depth[visible][onscreen]), 0.3, 8.0)
    ax.scatter(
        px[onscreen], py[onscreen], s=sizes, c=[STAR_COLOR], alpha=0.7, linewidths=0, zorder=0
    )

    # --- cluster billboard (real MUSE image), placed at its true distance ---
    # Projected at the MUSE frame's own centre, not the scene origin, so the
    # photo lands exactly on top of the galaxies it contains.
    bb_depth = CLUSTER_D_C - cam_z
    bb_alpha = billboard_alpha(cam_z) if bb_depth > NEAR_CLIP else 0.0
    if bb_alpha > 0.01:
        half_w = np.radians(MUSE_W_ARCSEC / 3600.0 / 2.0) * CLUSTER_D_C
        half_h = np.radians(MUSE_H_ARCSEC / 3600.0 / 2.0) * CLUSTER_D_C
        bx = f * half_w / bb_depth
        by = f * half_h / bb_depth
        bcx = f * BILLBOARD_X / bb_depth
        bcy = f * BILLBOARD_Y / bb_depth
        if bx > 0.001:  # skip once absurdly small/far
            # feather the billboard's edge so it reads as a volume, not a
            # pasted rectangular card: fade alpha to 0 over the outer ~12%
            # of the image via a soft per-pixel alpha multiplier
            bh_img, bw_img = billboard_img.shape[:2]
            yy, xx = np.mgrid[0:bh_img, 0:bw_img]
            edge_frac = 0.12
            dist_to_edge = np.minimum(
                np.minimum(xx, bw_img - 1 - xx) / (bw_img * edge_frac),
                np.minimum(yy, bh_img - 1 - yy) / (bh_img * edge_frac),
            )
            edge_alpha = np.clip(dist_to_edge, 0, 1)
            billboard_rgba = np.dstack(
                [billboard_img[..., :3], edge_alpha * float(bb_alpha)]
            )
            ax.imshow(
                billboard_rgba,
                extent=(bcx - bx, bcx + bx, bcy - by, bcy + by),
                zorder=1,
                interpolation="bilinear",
                origin="lower",
            )

    # --- catalog objects: painter's algorithm, far to near ---
    depths = np.array([o.d_c - cam_z for o in objects])
    order = np.argsort(-depths)
    label_entries = []
    # Real angular size grows as ~1/depth, so very close objects would
    # otherwise balloon into an opaque wall filling the whole frame. Cap the
    # displayed size and dissolve the object's alpha out as it swells past
    # that cap, so passing close to something reads as flying *through* it
    # rather than slamming into a flat card.
    MAX_EXT = 1.6 * VIEW_HALF
    FADE_START_EXT = 0.9 * MAX_EXT
    FADE_END_EXT = 4.0 * MAX_EXT
    for idx in order:
        o = objects[idx]
        d = depths[idx]
        if d <= NEAR_CLIP:
            continue
        # objects whose light is already in the billboard fade in exactly as
        # it fades out, so the cluster is never drawn twice and never absent
        dissolve_alpha = (1.0 - bb_alpha) if o.dissolve else 1.0
        if dissolve_alpha <= 0.01:
            continue
        px = f * o.x / d
        py = f * o.y / d
        raw_ext_w = f * o.half_w_mpc / d
        raw_ext_h = f * o.half_h_mpc / d
        raw_ext = max(raw_ext_w, raw_ext_h)
        scale = min(raw_ext, MAX_EXT) / raw_ext if raw_ext > 0 else 1.0
        half_ext_w = raw_ext_w * scale
        half_ext_h = raw_ext_h * scale
        if px + half_ext_w < -VIEW_HALF * 1.3 or px - half_ext_w > VIEW_HALF * 1.3:
            continue
        if py + half_ext_h < -VIEW_HALF * 1.3 or py - half_ext_h > VIEW_HALF * 1.3:
            continue
        if max(half_ext_w, half_ext_h) < 0.0008:
            continue

        if raw_ext <= FADE_START_EXT:
            close_alpha = 1.0
        else:
            close_alpha = 1.0 - np.clip(
                (raw_ext - FADE_START_EXT) / (FADE_END_EXT - FADE_START_EXT), 0, 1
            )
        close_alpha *= dissolve_alpha
        if close_alpha <= 0.01:
            continue

        ax.imshow(
            o.rgba,
            extent=(px - half_ext_w, px + half_ext_w, py - half_ext_h, py + half_ext_h),
            zorder=3,
            interpolation="bilinear",
            alpha=float(close_alpha),
            origin="lower",
        )

        if o.kind == "lensed" and show_labels:
            alpha = _alpha_for_depth_fade(d, LABEL_FADE_IN_MPC) * close_alpha
            if alpha > 0.01:
                label_entries.append((o.source, px, py + half_ext_h + 0.03, o.z, alpha, d))

    # one label per source per frame - the nearest (largest-alpha) image of
    # each source wins, since multiple images of one source are the physics
    # highlight, not a reason to print the same tag three times
    best_by_source = {}
    for source, px, py, z, alpha, d in label_entries:
        if source not in best_by_source or alpha > best_by_source[source][3]:
            best_by_source[source] = (px, py, z, alpha)

    for source, (px, py, z, alpha) in best_by_source.items():
        ax.text(
            px,
            py,
            f"Source {source}\nz={z:.3f}",
            color=(1.0, 0.9, 0.6, alpha),
            fontsize=11,
            ha="center",
            va="bottom",
            zorder=5,
            family="monospace",
        )

    if show_hud:
        z_now = z_lookup(cam_z)
        # D_A = D_C / (1+z), exact in a flat cosmology. Worth knowing before
        # you read this off the screen: it is not monotonic. It climbs to
        # 1773 Mpc at t = 15.8 s (z = 1.605) and then falls back to 1394 Mpc
        # by the end, so 39% of the flight counts *down* even though the
        # camera never stops moving forward. That is the real behaviour of
        # angular diameter distance, not a bug, and D_C is printed underneath
        # partly so there is a monotonic number to read it against.
        d_a = cam_z / (1.0 + z_now)
        ax.text(
            -0.97,
            0.94,
            f"z   = {z_now:.3f}\nD_A = {d_a:5.0f} Mpc\nD_C = {cam_z:5.0f} Mpc",
            color=(0.8, 1.0, 0.9),
            fontsize=12,
            ha="left",
            va="top",
            zorder=10,
            family="monospace",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="render a few still frames only")
    parser.add_argument("--preview-frames", type=int, default=8)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--no-hud", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--fig-size", type=float, default=None, help="figure size in inches")
    args = parser.parse_args()

    show_labels = SHOW_LABELS and not args.no_labels
    show_hud = SHOW_HUD and not args.no_hud

    field_rows, lensed_rows, stamps = load_scene()
    objects = build_objects(field_rows, lensed_rows, stamps)
    starfield = build_starfield()
    billboard_img = plt.imread(os.path.join(OUT_DIR, "muse_rgb.png"))
    z_lookup = build_redshift_lookup()

    print(f"Loaded {len(objects)} objects "
          f"({sum(o.kind == 'field' for o in objects)} field, "
          f"{sum(o.kind == 'lensed' for o in objects)} lensed images)")

    if args.preview:
        fig_size = args.fig_size or 6.0
        dpi = args.dpi or 110
        n = args.preview_frames
        out_dir = os.path.join(OUT_DIR, "preview_frames")
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi, facecolor="black")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        frame_indices = np.linspace(0, N_FRAMES - 1, n).astype(int)
        for k, i in enumerate(frame_indices):
            cam_z = camera_z_of_frame(i, N_FRAMES)
            render_frame(ax, cam_z, objects, starfield, billboard_img, z_lookup, show_labels, show_hud)
            out_path = os.path.join(out_dir, f"frame_{k:03d}.png")
            fig.savefig(out_path, facecolor="black")
            print(f"  saved {out_path}  (z={z_lookup(cam_z):.3f}, cam_z={cam_z:.0f} Mpc)")
        plt.close(fig)
        return

    import imageio

    fig_size = args.fig_size or 7.2
    dpi = args.dpi or 160
    out_path = args.out or os.path.join(OUT_DIR, "carousel_flythrough.mp4")
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi, facecolor="black")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for i in range(N_FRAMES):
            cam_z = camera_z_of_frame(i, N_FRAMES)
            render_frame(ax, cam_z, objects, starfield, billboard_img, z_lookup, show_labels, show_hud)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
            if i % 30 == 0:
                print(f"  frame {i}/{N_FRAMES}  z={z_lookup(cam_z):.3f}")
    finally:
        writer.close()
        plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
