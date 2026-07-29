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

from prepare_data import load_field_catalog, load_lensed_sources, COSMO

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

# Cluster billboard true footprint (see prepare_imagery: 340x348 px @ 0.2"/px)
MUSE_W_ARCSEC = 340 * 0.2
MUSE_H_ARCSEC = 348 * 0.2
CLUSTER_D_C = COSMO.comoving_distance(0.4895).value

STAR_COLOR = (1.0, 1.0, 0.95)


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
    __slots__ = ("x", "y", "d_c", "z", "rgba", "half_w_mpc", "half_h_mpc", "kind", "label", "source")

    def __init__(self, x, y, d_c, z, rgba, half_w_mpc, half_h_mpc, kind, label, source=None):
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
            Object3D(r["x"], r["y"], r["d_c"], r["z"], s["rgba"], half_w, half_h, "field", r["label"])
        )
    for r in lensed_rows:
        s = stamps.get(f"img_{r['label']}")
        if s is None:
            continue
        half_w = object_angular_halfsize_deg(r["d_c"], s["half_width_arcsec"])
        half_h = object_angular_halfsize_deg(r["d_c"], s["half_height_arcsec"])
        objects.append(
            Object3D(
                r["x"], r["y"], r["d_c"], r["z"], s["rgba"], half_w, half_h,
                "lensed", r["label"], r["source"],
            )
        )
    return objects


NEAR_CLIP = 3.0  # Mpc; objects closer than this to the camera are culled
VIEW_HALF = 1.0  # normalized view-space half-extent (matches xlim/ylim)
LABEL_FADE_IN_MPC = 500.0
LABEL_FADE_OUT_MPC = 220.0


def _alpha_for_depth_fade(depth, fade_in, fade_out):
    """1.0 while approaching within fade_in Mpc, ramping down to 0 over
    fade_out Mpc after passing (depth goes negative once behind the object's
    own plane isn't quite right here -- depth is always > NEAR_CLIP for
    visible objects, so 'after passing' means depth is small)."""
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
    bb_depth = CLUSTER_D_C - cam_z
    bb_fade = 0.0
    if bb_depth > NEAR_CLIP:
        half_w = np.radians(MUSE_W_ARCSEC / 3600.0 / 2.0) * CLUSTER_D_C
        half_h = np.radians(MUSE_H_ARCSEC / 3600.0 / 2.0) * CLUSTER_D_C
        bx_raw = f * half_w / bb_depth
        by_raw = f * half_h / bb_depth
        fade = 1.0 - np.clip((bx_raw - 3.0 * VIEW_HALF) / (6.0 * VIEW_HALF), 0, 1)
        bx = min(bx_raw, 4.0 * VIEW_HALF)
        by = by_raw * (bx / bx_raw) if bx_raw > 0 else by_raw
        if bx > 0.001 and fade > 0.01:  # skip once absurdly small/far/faded
            bb_fade = fade
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
                [billboard_img[..., :3], edge_alpha * float(bb_fade)]
            )
            ax.imshow(
                billboard_rgba,
                extent=(-bx, bx, -by, by),
                zorder=1,
                interpolation="bilinear",
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
        if o.kind == "field" and bb_depth > NEAR_CLIP and bb_fade > 0.05:
            # cluster members are already visible in the billboard photo at
            # essentially the same depth; only render them individually once
            # the billboard has faded, so the cluster isn't drawn twice
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
        if close_alpha <= 0.01:
            continue

        if o.kind == "lensed":
            # soft warm-gold glow rim behind the real stamp - a subtle
            # accent, not a dominant flat disc, now that the stamp itself
            # carries the real arc shape
            glow_ext = max(half_ext_w, half_ext_h) * 1.15
            glow_alpha = np.clip(0.35 * (150.0 / max(d, 30.0)), 0.04, 0.22) * close_alpha
            ax.add_patch(
                plt.Circle(
                    (px, py),
                    glow_ext,
                    color=(1.0, 0.82, 0.35),
                    alpha=glow_alpha,
                    zorder=2,
                    linewidth=0,
                )
            )

        ax.imshow(
            o.rgba,
            extent=(px - half_ext_w, px + half_ext_w, py - half_ext_h, py + half_ext_h),
            zorder=3,
            interpolation="bilinear",
            alpha=float(close_alpha),
        )

        if o.kind == "lensed" and show_labels:
            alpha = _alpha_for_depth_fade(d, LABEL_FADE_IN_MPC, LABEL_FADE_OUT_MPC) * close_alpha
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
            fontsize=7,
            ha="center",
            va="bottom",
            zorder=5,
            family="monospace",
        )

    if show_hud:
        z_now = z_lookup(cam_z)
        ax.text(
            -0.97,
            0.94,
            f"z = {z_now:.3f}\nD_C = {cam_z:6.0f} Mpc",
            color=(0.8, 1.0, 0.9),
            fontsize=9,
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
